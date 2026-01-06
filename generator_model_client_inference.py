#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import requests
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--description", type=str, required=True, help="Description of the UI to generate")
parser.add_argument("--model", type=str, default="rldfcoder-qwen3-sketch", help="Model name registered on server")
parser.add_argument("--url", type=str, default="http://localhost:11434/api/generate", help="Hosted API endpoint")
args = parser.parse_args()

# Prompt template for UI generation
prompt_prefix = "provide the complete HTML code for a web page implemented with only tailwind CSS and font awesome icons. do not use any templating languages like jinja. the result should resemble an award-winning iOS app. include realistic and complete placeholder data. do not treat this as the starting point for an app - it should be the mockup of a final complete UI. remember to include alt text for all images. do not use javascript. do not use SVGs. here is a description of the webpage: "

# Construct full prompt
full_prompt = prompt_prefix + args.description

# Prepare request data
data = {
    "model": args.model,
    "stream": False,
    "prompt": full_prompt,
    "options": {"num_predict": 4096}
}

# Send request to server
print(f"Generating UI for: {args.description}")
print("Waiting for response...\n")

response = requests.post(args.url, json=data)

if response.status_code == 200:
    result = response.json()["response"]
    print("Generated HTML:")
    print("="*80)
    print(result)
    print("="*80)
else:
    print(f"Error: {response.status_code}")
    print(response.text)
