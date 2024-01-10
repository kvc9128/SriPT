import xml.etree.ElementTree as ET
import re
import os


def extract_body_text(sgm_content):
	try:
		# Attempt to parse using ElementTree
		root = ET.fromstring('<ROOT>' + sgm_content + '</ROOT>')  # Wrap with a root tag
		body_texts = [elem.text for elem in root.findall('.//BODY')]
		return body_texts
	except ET.ParseError:
		# Fallback to regex if XML parsing fails
		return re.findall(r'<BODY>(.*?)</BODY>', sgm_content, re.DOTALL)


def process_sgm_files(file_list, output_file_path):
	with open(output_file_path, 'w', encoding='utf-8') as output_file:
		for file_path in file_list:
			if os.path.exists(file_path):
				try:
					with open(file_path, 'r', encoding='utf-8') as file:
						sgm_content = file.read()
						body_texts = extract_body_text(sgm_content)
						for text in body_texts:
							output_file.write(text + '\n\n')  # Add a newline for separation
				except:
					print(f"Error reading file {file_path}, skipping and moving on")
			else:
				print(f'File not found: {file_path}')


sgm_file_list = [
	"Datasets/Reuters/reut2-000.sgm",
	"Datasets/Reuters/reut2-001.sgm",
	"Datasets/Reuters/reut2-002.sgm",
	"Datasets/Reuters/reut2-003.sgm",
	"Datasets/Reuters/reut2-004.sgm",
	"Datasets/Reuters/reut2-005.sgm",
	"Datasets/Reuters/reut2-006.sgm",
	"Datasets/Reuters/reut2-007.sgm",
	"Datasets/Reuters/reut2-008.sgm",
	"Datasets/Reuters/reut2-009.sgm",
	"Datasets/Reuters/reut2-010.sgm",
	"Datasets/Reuters/reut2-011.sgm",
	"Datasets/Reuters/reut2-012.sgm",
	"Datasets/Reuters/reut2-013.sgm",
	"Datasets/Reuters/reut2-014.sgm",
	"Datasets/Reuters/reut2-015.sgm",
	"Datasets/Reuters/reut2-016.sgm",
	"Datasets/Reuters/reut2-017.sgm",
	"Datasets/Reuters/reut2-018.sgm",
	"Datasets/Reuters/reut2-020.sgm",
	"Datasets/Reuters/reut2-021.sgm"
]  # Replace with your actual file paths
output_file_path = 'REUTERS_NEWS.txt'

# Process the list of SGM files
process_sgm_files(sgm_file_list, output_file_path)

print(f'Extracted text written to {output_file_path}')
