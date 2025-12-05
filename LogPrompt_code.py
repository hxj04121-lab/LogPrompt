import pandas as pd
import textwrap
from typing import List
from tqdm import tqdm
import time
import re
import warnings
import requests
import json
import argparse
import numpy as np
import yaml
import os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
parser=argparse.ArgumentParser()
parser.add_argument('--API_KEY',type=str)#Specify your API key here
parser.add_argument('--dataset',type=str)#excel file path, with column 'log' containing rows of raw logs
parser.add_argument('--strategy',type=str)#prompt strategies, choice between [Self,CoT,InContext]
parser.add_argument('--output_file_name',type=str,default="result.xlsx")
parser.add_argument('--example_file',type=str,default='')# example file for the in-context prompt, a excel file with two columns: [log, label]. The label column should be "normal" or "abnormal".
parser.add_argument('--API_BASE',type=str,default="https://api.openai.com/v1")#Specify your API base URL here
parser.add_argument('--API_URL',type=str,default=None)#Specify full API URL (overrides API_BASE)
parser.add_argument('--limit',type=int,default=None)#Limit the number of logs to process


args=parser.parse_args()

# Load config from yaml if exists
config = {}
if os.path.exists('config.yaml'):
    with open('config.yaml', 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(f"Error reading config.yaml: {exc}")

# Determine API KEY - support both uppercase and lowercase config keys
if args.API_KEY:
    OPENAI_API_KEY = args.API_KEY
elif config:
    # Try both uppercase and lowercase keys
    OPENAI_API_KEY = config.get('OPENAI_API_KEY') or config.get('api_key') or config.get('API_KEY')
    if not OPENAI_API_KEY:
        OPENAI_API_KEY = None
        print("Warning: No API Key found in config.yaml (looked for: OPENAI_API_KEY, api_key, API_KEY)")
else:
    OPENAI_API_KEY = None
    print("Warning: No API Key provided in args or config.yaml")

# Determine API URL - support both full URL and base URL
API_URL = None
API_BASE = None

if args.API_URL:
    # User provided full API URL via command line
    API_URL = args.API_URL
elif config:
    # Try to get full URL from config (support both uppercase and lowercase)
    API_URL = config.get('API_URL') or config.get('api_url') or config.get('API_url')
    if not API_URL:
        # Build from API_BASE (support both uppercase and lowercase)
        API_BASE = config.get('API_BASE') or config.get('api_base') or config.get('API_base')
        if not API_BASE:
            API_BASE = args.API_BASE

if not API_URL and not API_BASE:
    # Fall back to argument
    API_BASE = args.API_BASE

# Build API URL if not already set from full URL
if not API_URL and API_BASE:
    # Build API URL - handle different proxy formats
    if API_BASE.endswith('/'):
        API_BASE = API_BASE.rstrip('/')
    
    if API_BASE.endswith('/v1'):
        API_URL = f"{API_BASE}/chat/completions"
    elif '/v1/' in API_BASE:
        API_URL = f"{API_BASE}/chat/completions" if not API_BASE.endswith('/') else f"{API_BASE}chat/completions"
    elif 'openai.com' in API_BASE or 'openai' in API_BASE.lower():
        # Standard OpenAI format
        if not API_BASE.endswith('/v1'):
            API_URL = f"{API_BASE}/v1/chat/completions"
        else:
            API_URL = f"{API_BASE}/chat/completions"
    else:
        # For proxy services, assume /v1 is needed if not present
        if not API_BASE.endswith('/v1'):
            API_URL = f"{API_BASE}/v1/chat/completions"
        else:
            API_URL = f"{API_BASE}/chat/completions"

if not API_URL:
    raise Exception("Could not determine API URL. Please set API_URL or API_BASE in config.yaml or via command line arguments.")

print(f"Using API URL: {API_URL}")
# OPENAI_API_KEY is already set above
INPUT_FILE=args.dataset
PROMPT_STRATEGIES=args.strategy
OUTPUT_FILE=args.output_file_name
EXAMPLE_FILE=args.example_file
warnings.simplefilter(action='ignore', category=FutureWarning)
def filter_special_chars_for_F1(s):
    special_chars = r'[^\w\s*]'
    filtered_str = re.sub(special_chars, '', s)
    return filtered_str
def filter_special_characters(input_string):
    return re.sub(r'[^\w\s]', '', input_string).replace('true','').replace('false','')

def generate_prompt(prompt_header,logs: List[str],max_len=1000,no_reason=False) -> List[str]:
    prompt_parts_count=[]
    prompt_parts = []
    prompt=prompt_header
    log_count=0
    startStr=""
    for i, log in enumerate(logs):
        if no_reason:
            startStr+="("+str(i+1)+"x\n"
        else:
            startStr+="("+str(i+1)+"x-y\n"
        log_str = f"({i+1}) {log}"
        log_length = len(log_str)
        prompt_length=len(prompt)
        if log_length > max_len:
            print("warning: this log is too long")

        if prompt_length + log_length <= max_len:
            prompt += f" {log_str}"
            prompt_length += log_length + 1
            log_count+=1
            if i<(len(logs)-1) and (prompt_length+len(logs[i+1]))>=max_len:
                prompt_parts.append(prompt.replace("!!FormatControl!!",startStr).replace("!!NumberControl!!",str(log_count)))
                prompt_parts_count.append(log_count)
                log_count=0
                prompt=prompt_header
                startStr=""
                continue
            if i== (len(logs)-1):
                prompt_parts.append(prompt.replace("!!FormatControl!!",startStr).replace("!!NumberControl!!",str(log_count)))
                prompt_parts_count.append(log_count)
        else:
            if prompt!=prompt_header:
                log_count+=1
                prompt+=f" {log_str}"
                prompt_parts.append(prompt.replace("!!FormatControl!!",startStr).replace("!!NumberControl!!",str(log_count)))
                prompt_parts_count.append(log_count)
            else:
                prompt=prompt.replace("!!FormatControl!!",startStr)
                prompt=f"{prompt} ({i+1}) {log}"
                prompt_parts.append(prompt)
                prompt_parts_count.append(1)
            log_count=0
            prompt=prompt_header
            startStr=""
    return prompt_parts,prompt_parts_count

def filter_numbers(text):
    pattern = r'\(\d+\)'
    return re.sub(pattern, '', text)

def normalize_prediction(pred_text):
    """Convert prediction text to binary label (0=normal, 1=abnormal)"""
    if not pred_text or pd.isna(pred_text):
        return None
    
    pred_lower = str(pred_text).lower().strip()
    
    # Check for abnormal indicators
    if any(keyword in pred_lower for keyword in ['abnormal', 'abnor', '1', 'error', 'alert', 'interrupt', 'exception']):
        # Make sure it's not "normal" that contains these substrings
        if 'normal' not in pred_lower or pred_lower.startswith('abnormal'):
            return 1
    
    # Check for normal indicators
    if any(keyword in pred_lower for keyword in ['normal', 'norm', '0']):
        return 0
    
    # Default: if contains "abnormal" anywhere, it's abnormal
    if 'abnormal' in pred_lower:
        return 1
    
    # Default to normal if unclear
    return 0

def calculate_metrics(y_true, y_pred):
    """Calculate precision, recall, F1, and accuracy"""
    # Filter out None predictions
    valid_indices = [i for i, pred in enumerate(y_pred) if pred is not None]
    if len(valid_indices) == 0:
        return None
    
    y_true_valid = [y_true[i] for i in valid_indices]
    y_pred_valid = [y_pred[i] for i in valid_indices]
    
    try:
        precision = precision_score(y_true_valid, y_pred_valid, zero_division=0)
        recall = recall_score(y_true_valid, y_pred_valid, zero_division=0)
        f1 = f1_score(y_true_valid, y_pred_valid, zero_division=0)
        accuracy = accuracy_score(y_true_valid, y_pred_valid)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'valid_samples': len(valid_indices),
            'total_samples': len(y_true)
        }
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return None
def reprompt(raw_file_name,j,df_raw_answer,temperature):
    URL=API_URL
    headers={'Content-Type':'application/json','Authorization':f'Bearer {OPENAI_API_KEY}'}
    prompt=df_raw_answer.loc[j,"prompt"]
    msgs=[]
    payload={
        "model":"gpt-3.5-turbo",
        "temperature":temperature,
        "top_p":1,
        "n":1,
        "stream":False,
        "stop":None,
        "presence_penalty":0,
        "frequency_penalty":0,
        "logit_bias":{}
            }

    msgs.append({'role':"user","content":prompt})
    payload["messages"]=msgs
    parsed_log=""
    retry_count = 0
    max_retries = 3
    while parsed_log =='' and retry_count < max_retries:
        try:
            response=requests.request("POST",URL,headers=headers,data=json.dumps(payload), timeout=60)
            
            # Check response status
            if response.status_code != 200:
                print(f"HTTP Error {response.status_code}: {response.text[:500]}")
                retry_count += 1
                if retry_count >= max_retries:
                    raise Exception(f"HTTP {response.status_code} after {max_retries} retries")
                time.sleep(2)
                continue
            
            # Try to parse JSON response
            try:
                res=response.json()
            except json.JSONDecodeError as json_err:
                print(f"JSON Parse Error: {json_err}")
                print(f"Response text (first 500 chars): {response.text[:500]}")
                print(f"Response headers: {dict(response.headers)}")
                # Try to extract first valid JSON object if multiple objects exist
                text = response.text.strip()
                if text.startswith('{'):
                    # Try to find the first complete JSON object
                    brace_count = 0
                    end_pos = 0
                    for i, char in enumerate(text):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_pos = i + 1
                                break
                    if end_pos > 0:
                        try:
                            res = json.loads(text[:end_pos])
                            print("Successfully extracted first JSON object from response")
                        except:
                            raise Exception(f"Could not parse JSON: {json_err}")
                    else:
                        raise Exception(f"Could not find complete JSON object: {json_err}")
                else:
                    raise Exception(f"Response is not valid JSON: {json_err}")
            
            if "choices" not in res:
                if "error" in res:
                    error_code = res["error"].get("code", "unknown")
                    error_msg = res["error"].get("message", "Unknown error")
                    print(f"API Error [{error_code}]: {error_msg}")
                    if error_code in ["unsupported_country_region_territory", "invalid_api_key", "insufficient_quota"]:
                        raise Exception(f"Fatal API error: {error_code} - {error_msg}")
                else:
                    print("Unexpected response:", res)
                retry_count += 1
                if retry_count >= max_retries:
                    raise Exception("Max retries reached, API request failed")
                time.sleep(2)
                continue
            parsed_log=res["choices"][0]["message"]["content"]
        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries:
                raise Exception(f"Failed after {max_retries} retries: {str(e)}")
            print(f"Error (attempt {retry_count}/{max_retries}): {str(e)}")
            time.sleep(2)
    
    if parsed_log == '':
        raise Exception("Failed to get response from API after all retries")

    df_raw_answer.loc[j,"answer"]=parsed_log
    df_raw_answer.to_excel(raw_file_name,index=False)
    return parsed_log

def parse_logs(raw_file_name,prompt_parts: List[str],prompt_parts_count) -> List[str]:
    parsed_logs = []
    URL=API_URL
    headers={'Content-Type':'application/json','Authorization':f'Bearer {OPENAI_API_KEY}'}
    for p,prompt in tqdm(enumerate(prompt_parts)):
        msgs=[]
        payload={
            "model":"gpt-3.5-turbo",
            "temperature":0.5,
            "top_p":1,
            "n":1,
            "stream":False,
            "stop":None,
            "presence_penalty":0,
            "frequency_penalty":0,
            "logit_bias":{}
                }
        log_count=prompt_parts_count[p]
        msgs.append({'role':"user","content":prompt})
        payload["messages"]=msgs
        parsed_log=""
        retry_count = 0
        max_retries = 3
        print(f"\n处理批次 {p+1}/{len(prompt_parts)} (包含 {log_count} 条日志)...")
        
        while parsed_log =='' and retry_count < max_retries:
            try:
                print(f"  发送API请求 (尝试 {retry_count+1}/{max_retries})...")
                response=requests.request("POST",URL,headers=headers,data=json.dumps(payload), timeout=120)
                
                # Check response status
                if response.status_code != 200:
                    print(f"  HTTP Error {response.status_code}: {response.text[:500]}")
                    retry_count += 1
                    if retry_count >= max_retries:
                        raise Exception(f"HTTP {response.status_code} after {max_retries} retries")
                    time.sleep(2)
                    continue
                
                # Try to parse JSON response
                try:
                    res=response.json()
                except json.JSONDecodeError as json_err:
                    print(f"  JSON Parse Error: {json_err}")
                    print(f"  Response text (first 500 chars): {response.text[:500]}")
                    # Try to extract first valid JSON object if multiple objects exist
                    text = response.text.strip()
                    if text.startswith('{'):
                        # Try to find the first complete JSON object
                        brace_count = 0
                        end_pos = 0
                        for i, char in enumerate(text):
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    end_pos = i + 1
                                    break
                        if end_pos > 0:
                            try:
                                res = json.loads(text[:end_pos])
                                print("  Successfully extracted first JSON object from response")
                            except:
                                retry_count += 1
                                if retry_count >= max_retries:
                                    raise Exception(f"Could not parse JSON after {max_retries} retries: {json_err}")
                                print(f"  Retrying... (attempt {retry_count+1}/{max_retries})")
                                time.sleep(2)
                                continue
                        else:
                            retry_count += 1
                            if retry_count >= max_retries:
                                raise Exception(f"Could not find complete JSON object: {json_err}")
                            print(f"  Retrying... (attempt {retry_count+1}/{max_retries})")
                            time.sleep(2)
                            continue
                    else:
                        retry_count += 1
                        if retry_count >= max_retries:
                            raise Exception(f"Response is not valid JSON: {json_err}")
                        print(f"  Retrying... (attempt {retry_count+1}/{max_retries})")
                        time.sleep(2)
                        continue
                
                if "choices" not in res:
                    if "error" in res:
                        error_code = res["error"].get("code", "unknown")
                        error_msg = res["error"].get("message", "Unknown error")
                        print(f"  API Error [{error_code}]: {error_msg}")
                        if error_code in ["unsupported_country_region_territory", "invalid_api_key", "insufficient_quota"]:
                            raise Exception(f"Fatal API error: {error_code} - {error_msg}")
                    else:
                        print(f"  Unexpected response: {res}")
                    retry_count += 1
                    if retry_count >= max_retries:
                        raise Exception("Max retries reached, API request failed")
                    time.sleep(2)
                    continue
                
                # Successfully got response
                parsed_log = res["choices"][0]["message"]["content"]
                if parsed_log:
                    print(f"  ✓ 成功获取响应 (长度: {len(parsed_log)} 字符)")
                    break  # Success, exit loop
                else:
                    print(f"  ⚠ 响应为空，重试...")
                    retry_count += 1
                    time.sleep(2)
                    
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"  ✗ 失败: {str(e)}")
                    raise Exception(f"Failed after {max_retries} retries: {str(e)}")
                print(f"  ⚠ 错误 (尝试 {retry_count}/{max_retries}): {str(e)}")
                time.sleep(2)
        
        if parsed_log == '':
            raise Exception("Failed to get response from API after all retries")
        
        parsed_logs.append(parsed_log)
    
    print(f"\n✓ 所有批次处理完成，共 {len(parsed_logs)} 个响应")
    pd.DataFrame(data=list(zip(prompt_parts,parsed_logs)),columns=['prompt','answer']).to_excel(raw_file_name)
    print(f"原始结果已保存到: {raw_file_name}")
    return parsed_logs

def extract_log_index(prompts):
    log_numbers=[]
    for prompt in prompts:
        # Try to extract log numbers from the actual logs section ("the logs begin:")
        try:
            if "the logs begin:" in prompt.lower() or "the logs begin" in prompt.lower():
                # Extract from the logs section
                logs_section = prompt.split("the logs begin:")[-1].split("There are")[0]
                log_number = re.findall(r'\((\d{1,4})\)', logs_section)
            else:
                # Fallback: try to find in format section or entire prompt
                log_number = re.findall(r'\((\d{1,4})\)', prompt)
            
            # Remove duplicates and sort
            log_number = sorted(list(set([int(x) for x in log_number])))
            
            # If still empty, try alternative pattern
            if not log_number:
                # Try to find format pattern like (1)x-y
                log_number = re.findall(r'\((\d{1,4})\)[x-z]', prompt)
                log_number = sorted(list(set([int(x) for x in log_number])))
                
            log_numbers.append(log_number)
        except Exception as e:
            print(f"Warning: Failed to extract log numbers from prompt: {e}")
            log_numbers.append([])
    return log_numbers

def write_to_excel(raw_file_name,df_raw_answer: pd.DataFrame, logs: List[str], original_df: pd.DataFrame = None) -> tuple:
    print(f"\n开始解析和对齐结果...")
    reprompt_num=0
    prompts=df_raw_answer['prompt'].tolist()
    log_numbers=extract_log_index(prompts)
    parsed_logs=df_raw_answer['answer'].tolist()
    parsed_logs_per_log = []
    for i, parsed_log in enumerate(parsed_logs):
        log_parts = parsed_log
        parsed_logs_per_log.append(log_parts)
        
    parsed_logs_df = pd.DataFrame()
    index=0
    print(f"需要解析 {len(parsed_logs_per_log)} 个响应，对应 {len(logs)} 条日志")
    for j, parsed_log in tqdm(enumerate(parsed_logs_per_log), desc="解析答案"):
        max_reprompts = 3
        reprompt_count = 0
        success = False
        while not success and reprompt_count < max_reprompts:
            temperature = 0.5 + reprompt_count * 0.4
            try:
                pattern = r"\({0}\).*?\({1}\)"
                xx_list=[]
                log_number=log_numbers[j] if j < len(log_numbers) else []
                
                # If no log numbers found in prompt, try to extract from answer
                if not log_number or len(log_number) == 0:
                    print(f"Warning: No log numbers found in prompt {j}, trying to extract from answer...")
                    # Try to extract log numbers from answer format
                    # Pattern 1: (数字字母-normal) or (数字字母-abnormal) - e.g., (70b-normal), (976x-normal)
                    answer_log_numbers = re.findall(r'\((\d{1,4})[a-z]-(normal|abnormal)\)', parsed_log, re.IGNORECASE)
                    if answer_log_numbers:
                        log_number = sorted(list(set([int(x[0]) for x in answer_log_numbers])))
                        print(f"Found {len(log_number)} log numbers in answer (format: number-letter-label): {log_number}")
                    else:
                        # Pattern 2: (数字字母-字母 单词: - e.g., (1044x-n Normal:
                        answer_log_numbers = re.findall(r'\((\d{1,4})[a-z]-[a-z]\s+(Normal|Abnormal):', parsed_log, re.IGNORECASE)
                        if answer_log_numbers:
                            log_number = sorted(list(set([int(x[0]) for x in answer_log_numbers])))
                            print(f"Found {len(log_number)} log numbers in answer (format: number-letter-letter-word): {log_number}")
                        else:
                            # Pattern 3: (数字字母-normal or (数字字母-abnormal (without closing paren)
                            answer_log_numbers = re.findall(r'\((\d{1,4})[a-z]-(normal|abnormal)\s+', parsed_log, re.IGNORECASE)
                            if answer_log_numbers:
                                log_number = sorted(list(set([int(x[0]) for x in answer_log_numbers])))
                                print(f"Found {len(log_number)} log numbers in answer (format: number-letter-label): {log_number}")
                            else:
                                # Pattern 4: (数字-normal: or (数字-abnormal: - format with colon
                                answer_log_numbers = re.findall(r'\((\d{1,4})-(normal|abnormal):', parsed_log, re.IGNORECASE)
                                if answer_log_numbers:
                                    log_number = sorted(list(set([int(x[0]) for x in answer_log_numbers])))
                                    print(f"Found {len(log_number)} log numbers in answer (format: number-label:): {log_number}")
                                else:
                                    # Pattern 5: (数字normal or (数字abnormal
                                    answer_log_numbers = re.findall(r'\((\d{1,4})(normal|abnormal)', parsed_log, re.IGNORECASE)
                                    if answer_log_numbers:
                                        log_number = sorted(list(set([int(x[0]) for x in answer_log_numbers])))
                                        print(f"Found {len(log_number)} log numbers in answer: {log_number}")
                                    else:
                                        # Pattern 6: (数字n or (数字a (short format)
                                        answer_log_numbers = re.findall(r'\((\d{1,4})[na]', parsed_log, re.IGNORECASE)
                                        if answer_log_numbers:
                                            log_number = sorted(list(set([int(x) for x in answer_log_numbers])))
                                            print(f"Found {len(log_number)} log numbers in answer (short format): {log_number}")
                                        else:
                                            # Pattern 7: Generic pattern (数字)
                                            answer_log_numbers = re.findall(r'\((\d{1,4})\)', parsed_log)
                                            if answer_log_numbers:
                                                log_number = sorted(list(set([int(x) for x in answer_log_numbers])))
                                                print(f"Found {len(log_number)} log numbers in answer (generic): {log_number}")
                
                if not log_number or len(log_number) == 0:
                    print(f"Error: Could not find log numbers for prompt {j}, skipping...")
                    break
                    
                # Parse answer format: (数字normal - ...) or (数字abnormal - ...)
                # Note: Format is (1normal - ...), NOT (1)normal - ...
                parsed_log_clean = parsed_log.replace('\n', ' ').replace('\r', ' ')
                
                # Debug: print log numbers and sample of answer
                if j == 0:  # Only print for first prompt to avoid spam
                    print(f"Debug: log_numbers[{j}] = {log_number}")
                    print(f"Debug: Answer preview (first 200 chars): {parsed_log_clean[:200]}")
                
                # Extract each log entry by finding start and end markers
                for i, log_idx in enumerate(log_number):
                    content = None
                    
                    # Try patterns in order of likelihood
                    # Pattern 1: (数字字母-normal) or (数字字母-abnormal) - e.g., (976x-normal)
                    pattern1 = rf"\({log_idx}[a-z]-(normal|abnormal)\)"
                    match1 = re.search(pattern1, parsed_log_clean, re.IGNORECASE)
                    if match1:
                        # Extract content after the closing parenthesis
                        start_pos = match1.end()
                    else:
                        # Pattern 2: (数字字母-单个字母 单词: - e.g., (1044x-n Normal:
                        pattern2 = rf"\({log_idx}[a-z]-[a-z]\s+(Normal|Abnormal):"
                        match2 = re.search(pattern2, parsed_log_clean, re.IGNORECASE)
                        if match2:
                            start_pos = match2.end()
                        else:
                            # Pattern 3: (数字字母-normal or (数字字母-abnormal (without closing paren)
                            pattern3 = rf"\({log_idx}[a-z]-(normal|abnormal)\s+"
                            match3 = re.search(pattern3, parsed_log_clean, re.IGNORECASE)
                            if match3:
                                start_pos = match3.end()
                            else:
                                # Pattern 4: (数字-normal: ...) or (数字-abnormal: ...) - format with colon
                                pattern4 = rf"\({log_idx}-(normal|abnormal):\s*"
                                match4 = re.search(pattern4, parsed_log_clean, re.IGNORECASE)
                                if match4:
                                    start_pos = match4.end()
                                else:
                                    # Pattern 5: (数字normal - ...) or (数字abnormal - ...) - alternative format
                                    pattern5 = rf"\({log_idx}(normal|abnormal)\s*-\s*"
                                    match5 = re.search(pattern5, parsed_log_clean, re.IGNORECASE)
                                    if match5:
                                        start_pos = match5.end()
                                    else:
                                        # Pattern 6: (数字)n-... or (数字)a-... (short format)
                                        pattern6 = rf"\({log_idx}([na])\s*[-:]"
                                        match6 = re.search(pattern6, parsed_log_clean, re.IGNORECASE)
                                        if match6:
                                            start_pos = match6.end()
                                        else:
                                            # Pattern 7: (数字)normal or (数字)abnormal (without dash)
                                            pattern7 = rf"\({log_idx}(normal|abnormal)\s*"
                                            match7 = re.search(pattern7, parsed_log_clean, re.IGNORECASE)
                                            if match7:
                                                start_pos = match7.end()
                                            else:
                                                # Pattern 8: (数字) followed by content (fallback)
                                                pattern8 = rf"\({log_idx}\)"
                                                match8 = re.search(pattern8, parsed_log_clean)
                                                if match8:
                                                    start_pos = match8.end()
                                                else:
                                                    # Only print warning for first few to avoid spam
                                                    if i < 3:
                                                        print(f"Warning: Could not find start marker for log {log_idx} in answer")
                                                        print(f"  Pattern tried: {pattern1}")
                                                        print(f"  Answer sample: {parsed_log_clean[:300]}")
                                                    continue
                    
                    # Find end position (start of next log entry or end of string)
                    if i < len(log_number) - 1:
                        next_idx = log_number[i + 1]
                        # Try to find next log entry - must match same format style
                        next_patterns = [
                            rf"\({next_idx}[a-z]-(normal|abnormal)\)",  # (数字字母-normal) or (数字字母-abnormal)
                            rf"\({next_idx}[a-z]-[a-z]\s+(Normal|Abnormal):",  # (数字字母-字母 单词:
                            rf"\({next_idx}[a-z]-(normal|abnormal)\s+",  # (数字字母-normal or (数字字母-abnormal
                            rf"\({next_idx}-(normal|abnormal):",  # (数字-normal: or (数字-abnormal:
                            rf"\({next_idx}(normal|abnormal)",  # (数字normal or (数字abnormal
                            rf"\({next_idx}([na])",  # (数字n or (数字a
                            rf"\({next_idx}\)",  # (数字) fallback
                        ]
                        end_pos = None
                        for next_pat in next_patterns:
                            match_next = re.search(next_pat, parsed_log_clean[start_pos:])
                            if match_next:
                                end_pos = start_pos + match_next.start()
                                break
                        
                        if end_pos:
                            content = parsed_log_clean[start_pos:end_pos].strip()
                        else:
                            content = parsed_log_clean[start_pos:].strip()
                    else:
                        # Last log entry - get everything until end
                        content = parsed_log_clean[start_pos:].strip()
                    
                    if content:
                        xx_list.append(content)
                        
                # Fallback: try original pattern matching if new method didn't work
                if len(xx_list) == 0 and len(log_number) > 1:
                    print("Fallback: trying original pattern matching...")
                for i in range(len(log_number)-1):
                    start=log_number[i]
                    end=log_number[i+1]
                    if start!=end-1:
                        continue
                    match=re.search(pattern.format(start,end),parsed_log.replace('\n',''))
                    if match:
                        xx=match.group().split(f"({start})")[1].split(f"({end})")[0].strip()
                        xx_list.append(xx)
                                
                last_log_number=log_number[-1]
                pattern_last=r"\({0}\).*".format(last_log_number)
                match=re.search(pattern_last,parsed_log.replace('\n',''))
                if match:
                    xx=match.group().split(f"({last_log_number})")[1].strip()
                    xx_list.append(xx)
                    
                if len(xx_list) == 0:
                    raise Exception(f"Could not extract any log results from parsed response")
                    
                for parsed_log_part in xx_list:
                    if parsed_log_part ==None or parsed_log_part=="":
                        continue
                    pred_raw=filter_numbers(parsed_log_part.replace('<*>','')).strip(' ')
                    pred_label=pred_raw
                    if index < len(logs):
                        # Use pd.concat instead of append (pandas 2.0+ deprecated append)
                        new_row = pd.DataFrame([{'log':logs[index],'pred':pred_label}])
                        parsed_logs_df = pd.concat([parsed_logs_df, new_row], ignore_index=True)
                    index+=1
                success = True
            except Exception as e:
                reprompt_count += 1
                if reprompt_count >= max_reprompts:
                    print(f"Error: Failed to parse after {max_reprompts} attempts: {e}")
                    print(f"Skipping prompt {j}")
                    break
                print(f"{e} - reprompting (attempt {reprompt_count}/{max_reprompts})...")
                try:
                    parsed_log=reprompt(raw_file_name,j,df_raw_answer,temperature)
                except Exception as reprompt_error:
                    print(f"Reprompt failed: {reprompt_error}")
                    break
    
    # Add ground truth labels if available
    df_for_labels = original_df if original_df is not None else None
    if df_for_labels is None:
        # Try to read from global df if available (fallback)
        try:
            df_for_labels = df
        except:
            pass
    
    if df_for_labels is not None and ('label' in df_for_labels.columns or 'Label' in df_for_labels.columns):
        label_col = 'label' if 'label' in df_for_labels.columns else 'Label'
        # Get labels for the processed logs (in same order)
        # Match logs to original dataframe by content
        labels = []
        for log in logs[:len(parsed_logs_df)]:
            # Find matching row in original dataframe
            matches = df_for_labels[df_for_labels['log'] == log] if 'log' in df_for_labels.columns else df_for_labels[df_for_labels['Content'] == log]
            if len(matches) > 0:
                labels.append(matches.iloc[0][label_col])
            else:
                labels.append(None)
        
        # Fallback: if matching failed, try direct indexing
        if len([l for l in labels if l is not None]) == 0:
            labels = df_for_labels[label_col].tolist()[:len(parsed_logs_df)]
        parsed_logs_df['true_label'] = labels[:len(parsed_logs_df)]
        
        # Normalize predictions to binary (0=normal, 1=abnormal)
        parsed_logs_df['pred_label'] = parsed_logs_df['pred'].apply(normalize_prediction)
        
        # Calculate metrics
        y_true = parsed_logs_df['true_label'].tolist()
        y_pred = parsed_logs_df['pred_label'].tolist()
        
        metrics = calculate_metrics(y_true, y_pred)
        
        if metrics:
            print("\n" + "="*60)
            print("实验指标 (Evaluation Metrics)")
            print("="*60)
            print(f"准确率 (Accuracy):  {metrics['accuracy']:.4f}")
            print(f"精确率 (Precision): {metrics['precision']:.4f}")
            print(f"召回率 (Recall):    {metrics['recall']:.4f}")
            print(f"F1分数 (F1-Score):  {metrics['f1']:.4f}")
            print(f"有效样本数:         {metrics['valid_samples']}/{metrics['total_samples']}")
            print("="*60)
            
            # Detailed classification report
            print("\n详细分类报告:")
            print(classification_report(y_true, y_pred, target_names=['Normal', 'Abnormal'], zero_division=0))
            
            # Save metrics to file
            metrics_file = raw_file_name.replace('.xlsx', '_metrics.txt')
            with open(metrics_file, 'w', encoding='utf-8') as f:
                f.write("实验指标 (Evaluation Metrics)\n")
                f.write("="*60 + "\n")
                f.write(f"准确率 (Accuracy):  {metrics['accuracy']:.4f}\n")
                f.write(f"精确率 (Precision): {metrics['precision']:.4f}\n")
                f.write(f"召回率 (Recall):    {metrics['recall']:.4f}\n")
                f.write(f"F1分数 (F1-Score):  {metrics['f1']:.4f}\n")
                f.write(f"有效样本数:         {metrics['valid_samples']}/{metrics['total_samples']}\n")
                f.write("="*60 + "\n\n")
                f.write("详细分类报告:\n")
                f.write(classification_report(y_true, y_pred, target_names=['Normal', 'Abnormal'], zero_division=0))
            print(f"\n指标已保存到: {metrics_file}")
        else:
            print("警告: 无法计算指标，可能预测结果格式不正确")
    
    parsed_logs_df.to_excel('Aligned_'+raw_file_name,index=False)
    print(f"\n结果已保存到: Aligned_{raw_file_name}")
    
if __name__ == "__main__":
    if INPUT_FILE.endswith('.csv'):
        df = pd.read_csv(INPUT_FILE)
    else:
        df = pd.read_excel(INPUT_FILE)
    
    # Ensure 'Content' column exists (handle 'log' alias)
    if 'Content' not in df.columns and 'log' in df.columns:
        df['Content'] = df['log']
    
    # Convert Label column to binary format if needed
    # For BGL dataset: '-' = normal (0), other labels = abnormal (1)
    if 'Label' in df.columns:
        # Convert to binary: '-' or empty = 0 (normal), others = 1 (abnormal)
        df['label'] = df['Label'].apply(lambda x: 0 if str(x).strip() == '-' or str(x).strip() == '' else 1)
    elif 'label' in df.columns:
        # Already in binary format, ensure it's numeric
        df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)
        
    np.random.seed(123)
    if PROMPT_STRATEGIES == 'CoT':
        # df=df.sample(frac=1).reset_index(drop=True) # Removed random sampling to keep order for small test
        answer_desc="a binary choice between normal and abnormal"
        prompt_header="Classify the given log entries into normal an abnormal categories. Do it with these steps: \
        (a) Mark it normal when values (such as memory address, floating number and register value) in a log are invalid. \
        (b) Mark it normal when lack of information. (c) Never consider <*> and missing values as abnormal patterns. \
        (d) Mark it abnormal when and only when the alert is explicitly expressed in textual content (such as keywords like error or interrupt). \
        Concisely explain your reason for each log. Organize your answer to be the following format: !!FormatControl!!, where x is %s and y is the reason. \
        There are !!NumberControl!! logs, the logs begin: "%(answer_desc)
        logs=df['Content'].tolist() # Use Content column from our structured CSV
        if args.limit:
             logs = logs[:args.limit]
             print(f"Running limited test with {len(logs)} logs.")
        else:
             print(f"Running full dataset with {len(logs)} logs.")

        ########### generate prompts ######################
        prompt_parts,prompt_parts_count = generate_prompt(prompt_header,logs,max_len=3000)
        ########### obtain raw answers from GPT ###########
        parse_logs = parse_logs(OUTPUT_FILE,prompt_parts,prompt_parts_count)
        ########### Align each log with its results #######
        df_raw_answer = pd.read_excel(OUTPUT_FILE)
        write_to_excel(OUTPUT_FILE,df_raw_answer,logs,df)
    if PROMPT_STRATEGIES == 'InContext':
        df_examples=pd.read_excel(EXAMPLE_FILE)
        df=df.sample(frac=1).reset_index(drop=True)
        answer_desc="a binary choice between 0 and 1"
        examples=' '.join(["(%d) Log: %s. Category: %s"%(i+1,df_examples.loc[i,'log'],int(df_examples.loc[i,'label']=='abnormal')) for i in range(len(df_examples))])
        prompt_header="Classify the given log entries into 0 and 1 categories based on semantic similarity to the following labelled example logs: %s.\
        Organize your answer to be the following format: !!FormatControl!!, where x is %s. There are !!NumberControl!! logs, the logs begin: "%(examples,answer_desc)
        logs=df['Content'].tolist() # Use Content column
        if args.limit:
             logs = logs[:args.limit]
             print(f"Running limited test with {len(logs)} logs.")
        else:
             print(f"Running full dataset with {len(logs)} logs.")
        ########### generate prompts ######################
        prompt_parts,prompt_parts_count = generate_prompt(prompt_header,logs,max_len=3000,no_reason=True)
        ########### obtain raw answers from GPT ###########
        parse_logs = parse_logs(OUTPUT_FILE,prompt_parts,prompt_parts_count)
        ########### Align each log with its results #######
        df_raw_answer = pd.read_excel(OUTPUT_FILE)
        write_to_excel(OUTPUT_FILE,df_raw_answer,logs,df)        
    if PROMPT_STRATEGIES == "Self":
        #candidate selection
        # df=df[:100] # Removed hard slicing here, controlling via logs list later
        prompt_candidates=[]
        with open('prompt_candidates.txt') as f:
            for line in f.readlines():
                prompt_candidates.append(line.strip('\n'))
        for i,prompt_candidate in tqdm(enumerate(prompt_candidates)):
            print('prompt %d'%(i+1))
            answer_desc="a parsed log template"
            prompt_header = "%s Organize your answer to be the following format: !!FormatControl!!, where x is %s. There are !!NumberControl!! logs, the logs begin: "%(prompt_candidate,answer_desc)
            logs=df['Content'].tolist() # Use Content column
            if args.limit and len(logs) > args.limit:
                 logs = logs[:args.limit]
                 print(f"Running limited test with {len(logs)} logs.")
            elif len(logs) > 50:
                 logs = logs[:50]
                 print(f"Running limited test with {len(logs)} logs.")
            ########### generate prompts ######################
            prompt_parts,prompt_parts_count = generate_prompt(prompt_header,logs,max_len=3000,no_reason=True)
            ########### obtain raw answers from GPT ###########
            parse_logs = parse_logs('Candidate_%d_'%(i+1)+OUTPUT_FILE,prompt_parts,prompt_parts_count)
            ########### Align each log with its results #######
            df_raw_answer = pd.read_excel(OUTPUT_FILE)
            write_to_excel('Candidate_%d_'%(i+1)+OUTPUT_FILE,df_raw_answer,logs,df)   
