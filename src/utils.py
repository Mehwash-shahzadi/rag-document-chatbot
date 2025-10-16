import os
import json
from typing import List, Tuple, Optional

def calculate_confidence(scores: List[float]) -> str:
    """
    Calculates confidence level from similarity scores.
    Args:
        scores (List[float]): List of similarity scores (0-1, higher is better).
    Returns:
        str: "High", "Medium", or "Low" confidence.
    """
    if not scores:
        return "Low"
    avg = sum(scores) / len(scores)
    if avg > 0.7:
        return "High"
    elif avg > 0.5:
        return "Medium"
    else:
        return "Low"

def format_response(answer: str, sources: str, confidence: str) -> str:
    """
    Formats chatbot response with markdown.
    Args:
        answer (str): The chatbot's answer.
        sources (str): Source citations.
        confidence (str): Confidence level.
    Returns:
        str: Markdown-formatted response.
    """
    return (
        f"**Answer:**\n{answer}\n\n"
        f"**Confidence:** {confidence}\n\n"
        f"**Sources:**\n{sources}"
    )

def save_feedback(question: str, answer: str, feedback: str, file_path: str = "feedback.json") -> None:
    """
    Saves user feedback to a JSON file.
    Args:
        question (str): User's question.
        answer (str): Chatbot's answer.
        feedback (str): 'up' or 'down'.
        file_path (str): Path to feedback JSON file.
    """
    entry = {
        "question": question,
        "answer": answer,
        "feedback": feedback
    }
    try:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = []
        data.append(entry)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error saving feedback: {e}")

def validate_file(file_path: str, allowed_types: Optional[List[str]] = None, max_size_mb: int = 10) -> Tuple[bool, Optional[str]]:
    """
    Validates file for size, type, and readability.
    Args:
        file_path (str): Path to file.
        allowed_types (Optional[List[str]]): Allowed file extensions (e.g., ['pdf', 'txt']).
        max_size_mb (int): Maximum allowed file size in MB.
    Returns:
        Tuple[bool, Optional[str]]: (Is valid, Error message if any)
    """
    if not os.path.isfile(file_path):
        return False, "File does not exist."
    if allowed_types:
        ext = os.path.splitext(file_path)[1][1:].lower()
        if ext not in allowed_types:
            return False, f"Unsupported file type: .{ext}"
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if size_mb > max_size_mb:
        return False, f"File size {size_mb:.2f} MB exceeds {max_size_mb} MB limit."
    try:
        with open(file_path, "rb") as f:
            f.read(1)
    except Exception as e:
        return False, f"File is not readable: {e}"
    return True, None