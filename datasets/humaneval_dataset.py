import re
import signal
import subprocess
import tempfile
from typing import Tuple


def humaneval_data_process(dataset):
    """Process HumanEval dataset into the required format."""
    list_data_dict = []
    for data in dataset:
        item = {
            "task": data.get("prompt", ""),
            "test": data.get("test", ""),
            "entry_point": data.get("entry_point", "")
        }
        list_data_dict.append(item)
    return list_data_dict


def humaneval_get_predict(response: str) -> str:
    """
    Extract Python code from the model's response.
    Handles markdown code blocks and plain code.
    """
    if not response:
        return ""
    
    # Try to extract code from markdown code blocks
    if "```python" in response:
        # Extract code between ```python and ```
        pattern = r"```python\n(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return matches[0].strip()
    elif "```" in response:
        # Extract code between ``` and ```
        pattern = r"```\n(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return matches[0].strip()
    
    # If no code blocks found, try to extract code after common prefixes
    if "Here is the solution:" in response or "Here's the solution:" in response:
        lines = response.split("\n")
        code_lines = []
        in_code = False
        for line in lines:
            if line.strip().startswith("def "):
                in_code = True
            if in_code:
                code_lines.append(line)
        if code_lines:
            return "\n".join(code_lines).strip()
    
    # If all else fails, look for function definitions
    if "def " in response:
        # Find the first function definition and extract everything from there
        idx = response.find("def ")
        if idx != -1:
            return response[idx:].strip()
    
    # Return the whole response as a last resort
    return response.strip()


def check_correctness(task_prompt: str, predicted_code: str, test_code: str, timeout: float = 3.0) -> Tuple[float, str]:
    """
    Check if the predicted code passes the test cases.
    
    Returns:
        (score, result_string): score is 1.0 if passed, 0.0 if failed
    """
    if not predicted_code or not predicted_code.strip():
        return 0.0, "No code generated"
    
    # Combine the task prompt (function signature), predicted code, and test code
    full_code = f"{task_prompt}\n{predicted_code}\n{test_code}\n\ncheck(candidate)"
    
    # Run the code in a subprocess with timeout
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(full_code)
            temp_file = f.name
        
        try:
            result = subprocess.run(
                ['python', temp_file],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode == 0:
                return 1.0, "Passed"
            else:
                error_msg = result.stderr or result.stdout or "Unknown error"
                return 0.0, f"Failed: {error_msg[:200]}"
        finally:
            import os
            if os.path.exists(temp_file):
                os.unlink(temp_file)
                
    except subprocess.TimeoutExpired:
        return 0.0, "Timeout"
    except Exception as e:
        return 0.0, f"Error: {str(e)[:200]}"

