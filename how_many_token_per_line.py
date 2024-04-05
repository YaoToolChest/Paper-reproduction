from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')



######################################txt
from transformers import AutoTokenizer

# Initialize the tokenizer (ensure you have the correct model for your task)

# Path to your input file
raw_file = 'filtered_output1.txt'
# Path to your output file
output_file = 'filtered_output.txt'

max_count_token = 0
max_token_line = ''
removed_lines_count = 0

with open(raw_file, 'r', encoding='utf-8') as file:
    lines = file.readlines()

filtered_lines = []

for line in lines:
    # Tokenize the line and count the number of tokens
    temp = len(tokenizer.tokenize(line))
    
    # Update max token count and line if current line has more tokens
    if temp > max_count_token:
        max_count_token = temp
        max_token_line = line.strip()  # .strip() removes trailing newline
    
    # If the line has more than 38 tokens, skip it and increment counter
    if temp > 38:
        removed_lines_count += 1
        continue  # Skip adding this line to filtered_lines
    
    # Add line to the filtered list
    filtered_lines.append(line)

# Write filtered lines to a new file
with open(output_file, 'w', encoding='utf-8') as file1:
    for line in filtered_lines:
        file1.write(line)

print(f'Max token count: {max_count_token}, in line: {max_token_line}')
print(f'Removed lines count: {removed_lines_count}')



#######################################csv
# import pandas as pd
# raw_csv = 'recurrent/Create_BaseData/City_name_V2.csv'
# raw_csv = pd.read_csv(raw_csv)
# token_counts = raw_csv['City'].apply(lambda x: len(tokenizer.encode(x, add_special_tokens=True)))

# # 找出最大的token数量
# max_token_count = token_counts.max()

# print(f"最大的token数量是: {max_token_count}")