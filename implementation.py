#!/usr/bin/env python3

import cv2
import torch
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class CaptionedSegment:
    """Data class for storing information about a captioned video segment."""
    timestamp: float
    frame_idx: int
    caption: str
    score: float
    frame: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation (without frame data)."""
        return {
            "timestamp": self.timestamp,
            "frame_idx": self.frame_idx,
            "caption": self.caption,
            "score": self.score
        }

class VideoCaptioner:
    """Generates captions for video frames using a BLIP model."""
    def __init__(self, 
                 model_name: str = "Salesforce/blip-image-captioning-base", 
                 device: str = None,
                 unconditional_prompt: str = None):
        """
        Initialize the video captioner.
        
        Args:
            model_name: Name or path of the BLIP captioning model
            device: Device to run the model on ('cuda', 'cpu', etc.)
            unconditional_prompt: Optional prompt to guide the captioning
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Initializing VideoCaptioner with model {model_name} on {self.device}")
        
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.unconditional_prompt = unconditional_prompt
        
    def caption_frame(self, frame: np.ndarray, beam_size: int = 5, max_length: int = 30) -> str:
        """
        Generate a caption for a single video frame.
        
        Args:
            frame: The video frame as a numpy array (BGR format from OpenCV)
            beam_size: Beam size for text generation
            max_length: Maximum length of the generated caption
            
        Returns:
            A string caption for the frame
        """
        # Convert BGR to RGB (BLIP expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process image
        if self.unconditional_prompt:
            inputs = self.processor(images=rgb_frame, text=self.unconditional_prompt, return_tensors="pt").to(self.device)
        else:
            inputs = self.processor(images=rgb_frame, return_tensors="pt").to(self.device)
        
        # Generate caption
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                num_beams=beam_size,
                max_length=max_length,
                min_length=5,
                early_stopping=True
            )
        
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption
    
    def __call__(self, frame: np.ndarray) -> str:
        """Allows using the captioner instance directly as a function."""
        return self.caption_frame(frame)


class RelevanceScorer:
    """Scores relevance between question and text (captions) using embedding similarity."""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = None):
        """
        Initialize the relevance scorer.
        
        Args:
            model_name: Name or path of the sentence transformer model
            device: Device to run the model on ('cuda', 'cpu', etc.)
        """
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Initializing RelevanceScorer with model {model_name} on {device}")
        self.embedder = SentenceTransformer(model_name, device=device)
    
    def score(self, question: str, caption: str) -> float:
        """
        Compute relevance score between question and caption.
        
        Args:
            question: The question text
            caption: The caption text
            
        Returns:
            A similarity score between 0 and 1
        """
        q_emb = self.embedder.encode(question, convert_to_tensor=True)
        c_emb = self.embedder.encode(caption, convert_to_tensor=True)
        return float(util.pytorch_cos_sim(q_emb, c_emb).item())
    
    def batch_score(self, question: str, captions: List[str]) -> List[float]:
        """
        Compute relevance scores for multiple captions in a batch.
        
        Args:
            question: The question text
            captions: List of caption texts
            
        Returns:
            List of similarity scores
        """
        q_emb = self.embedder.encode(question, convert_to_tensor=True)
        c_embs = self.embedder.encode(captions, convert_to_tensor=True)
        
        # Compute similarity for all captions at once
        scores = util.pytorch_cos_sim(q_emb.unsqueeze(0), c_embs)[0]
        return scores.cpu().numpy().tolist()
    
    def __call__(self, question: str, caption: str) -> float:
        """Allows using the scorer instance directly as a function."""
        return self.score(question, caption)


class AdaptiveFrameSampler:
    """Samples frames from a video using different strategies."""
    
    STRATEGIES = ["uniform", "scene_change", "hybrid"]
    
    def __init__(self, strategy: str = "hybrid", 
                 min_interval: int = 15,
                 scene_threshold: float = 30.0):
        """
        Initialize the frame sampler.
        
        Args:
            strategy: Sampling strategy ('uniform', 'scene_change', or 'hybrid')
            min_interval: Minimum interval between frames (in frames)
            scene_threshold: Threshold for scene change detection
        """
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Strategy must be one of {self.STRATEGIES}")
        
        self.strategy = strategy
        self.min_interval = min_interval
        self.scene_threshold = scene_threshold
        self.prev_frame = None
    
    def _detect_scene_change(self, curr_frame: np.ndarray) -> float:
        """
        Detect if there's a scene change between previous and current frame.
        
        Args:
            curr_frame: The current video frame
            
        Returns:
            Scene change score (higher means more change)
        """
        if self.prev_frame is None:
            self.prev_frame = curr_frame.copy()
            return float('inf')  # First frame is always selected
        
        # Convert to grayscale for faster processing
        prev_gray = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate difference
        diff = cv2.absdiff(prev_gray, curr_gray)
        score = np.mean(diff)
        
        # Update previous frame
        self.prev_frame = curr_frame.copy()
        
        return score
    
    def should_sample(self, frame: np.ndarray, frame_idx: int) -> bool:
        """
        Determine if the current frame should be sampled based on the strategy.
        
        Args:
            frame: The current video frame
            frame_idx: Index of the current frame
            
        Returns:
            True if the frame should be sampled, False otherwise
        """
        # Always sample the first frame
        if frame_idx == 0:
            self.prev_frame = frame.copy()
            return True
        
        # Enforce minimum interval
        if frame_idx % self.min_interval != 0:
            return False
        
        if self.strategy == "uniform":
            return True
        
        scene_score = self._detect_scene_change(frame)
        
        if self.strategy == "scene_change":
            return scene_score > self.scene_threshold
        
        # Hybrid strategy: sample at regular intervals but also at scene changes
        return scene_score > self.scene_threshold


class QuestionGuidedCaptioner:
    """
    Processes video frames to find segments most relevant to a given question.
    """
    def __init__(self, 
                 captioner: VideoCaptioner, 
                 scorer: RelevanceScorer,
                 sampler: Optional[AdaptiveFrameSampler] = None,
                 top_k: int = 5, 
                 min_score: float = 0.0,
                 keep_frames: bool = False):
        """
        Initialize the question-guided captioner.
        
        Args:
            captioner: VideoCaptioner instance for generating captions
            scorer: RelevanceScorer instance for scoring relevance
            sampler: AdaptiveFrameSampler instance for sampling frames (default: uniform sampling)
            top_k: Number of top-scoring segments to return
            min_score: Minimum score for a segment to be considered
            keep_frames: Whether to keep the frame data in the results
        """
        self.captioner = captioner
        self.scorer = scorer
        self.sampler = sampler or AdaptiveFrameSampler(strategy="uniform", min_interval=30)
        self.top_k = top_k
        self.min_score = min_score
        self.keep_frames = keep_frames
    
    def process_video(self, video_path: str, question: str) -> List[CaptionedSegment]:
        """
        Process the video to find segments relevant to the question.
        
        Args:
            video_path: Path to the video file
            question: The question text
            
        Returns:
            List of CaptionedSegment objects for the top-k most relevant segments
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        logger.info(f"Processing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video file {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video properties: {fps:.2f} fps, {total_frames} total frames")
        
        # Process frames
        candidates = []
        frame_idx = 0
        
        # Use tqdm for progress tracking
        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if self.sampler.should_sample(frame, frame_idx):
                    timestamp = frame_idx / fps
                    
                    # Generate caption
                    caption = self.captioner.caption_frame(frame)
                    
                    # Score relevance
                    rel_score = self.scorer.score(question, caption)
                    
                    # Create segment
                    segment = CaptionedSegment(
                        timestamp=timestamp,
                        frame_idx=frame_idx,
                        caption=caption,
                        score=rel_score,
                        frame=frame.copy() if self.keep_frames else None
                    )
                    
                    # Add to candidates if score is above threshold
                    if rel_score >= self.min_score:
                        candidates.append(segment)
                
                frame_idx += 1
                pbar.update(1)
        
        cap.release()
        
        logger.info(f"Generated {len(candidates)} candidate segments")
        
        # Sort by relevance (descending) and select top_k
        top_segments = sorted(candidates, key=lambda x: x.score, reverse=True)[:self.top_k]
        
        # Sort by timestamp for output
        top_segments.sort(key=lambda x: x.timestamp)
        
        return top_segments

    def save_results(self, segments: List[CaptionedSegment], output_dir: str, 
                     save_frames: bool = False, video_name: str = None) -> str:
        """
        Save results to output directory.
        
        Args:
            segments: List of CaptionedSegment objects
            output_dir: Directory to save results
            save_frames: Whether to save frame images
            video_name: Name of the video (for naming output files)
            
        Returns:
            Path to the saved results file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create base filename from video name or timestamp
        base_name = video_name or f"results_{int(time.time())}"
        
        # Save text results
        results_file = output_dir / f"{base_name}_results.txt"
        
        with open(results_file, 'w') as f:
            f.write(f"Top {len(segments)} relevant segments:\n\n")
            
            for i, seg in enumerate(segments, 1):
                mins = int(seg.timestamp // 60)
                secs = int(seg.timestamp % 60)
                f.write(f"{i}. [{mins:02d}:{secs:02d}] (Score: {seg.score:.3f})\n")
                f.write(f"   Caption: {seg.caption}\n\n")
        
        # Save frames if requested
        if save_frames:
            frames_dir = output_dir / f"{base_name}_frames"
            frames_dir.mkdir(exist_ok=True)
            
            for i, seg in enumerate(segments, 1):
                if seg.frame is not None:
                    frame_file = frames_dir / f"frame_{i:02d}_{int(seg.timestamp):04d}s.jpg"
                    cv2.imwrite(str(frame_file), seg.frame)
        
        logger.info(f"Results saved to {results_file}")
        return str(results_file)


def demo_pipeline():

    from argparse import ArgumentParser
    import time
    
    parser = ArgumentParser(description="Question-Guided Video Captioning Demo")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--question", type=str, required=True, help="Question to guide captioning")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top segments to return")
    parser.add_argument("--sample-strategy", type=str, default="hybrid", 
                        choices=AdaptiveFrameSampler.STRATEGIES, help="Frame sampling strategy")
    parser.add_argument("--sample-interval", type=int, default=30, help="Minimum frame sampling interval")
    parser.add_argument("--save-frames", action="store_true", help="Save frame images")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    
    args = parser.parse_args()
    
    # Initialize components
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info("Initializing pipeline components...")
    start_time = time.time()
    
    captioner = VideoCaptioner(
        model_name="Salesforce/blip-image-captioning-base", 
        device=device,
        unconditional_prompt="A detailed description of what is happening in this egocentric view:"
    )
    
    scorer = RelevanceScorer(
        model_name="all-MiniLM-L6-v2", 
        device=device
    )
    
    sampler = AdaptiveFrameSampler(
        strategy=args.sample_strategy,
        min_interval=args.sample_interval,
        scene_threshold=30.0  # Adjust based on your needs
    )
    
    pipeline = QuestionGuidedCaptioner(
        captioner=captioner,
        scorer=scorer,
        sampler=sampler,
        top_k=args.top_k,
        min_score=0.2,  # Discard very low-scoring segments
        keep_frames=args.save_frames
    )
    
    init_time = time.time() - start_time
    logger.info(f"Pipeline initialized in {init_time:.2f} seconds")
    
    # Process the video
    logger.info(f"Processing video with question: '{args.question}'")
    process_start = time.time()
    
    top_segments = pipeline.process_video(args.video, args.question)
    
    process_time = time.time() - process_start
    logger.info(f"Video processing completed in {process_time:.2f} seconds")
    
    # Save results
    if args.output_dir:
        video_name = Path(args.video).stem
        pipeline.save_results(top_segments, args.output_dir, args.save_frames, video_name)
    
    # Print results
    print(f"\nTop {len(top_segments)} segments for question: '{args.question}'")
    print("=" * 80)
    
    for i, seg in enumerate(top_segments, 1):
        mins = int(seg.timestamp // 60)
        secs = int(seg.timestamp % 60)
        print(f"{i}. [{mins:02d}:{secs:02d}] (Score: {seg.score:.3f})")
        print(f"   Caption: {seg.caption}")
        print()


if __name__ == "__main__":
    import time
    demo_pipeline()
