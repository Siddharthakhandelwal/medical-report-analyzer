import PyPDF2
from typing import List, Dict, Optional
import re

def extract_text_from_pdf(pdf_path: str) -> Dict[str, List[str]]:
    """
    Extract text from a PDF file in a structured way.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        Dict[str, List[str]]: Dictionary containing structured text with:
            - 'pages': List of text from each page
            - 'paragraphs': List of paragraphs from the entire document
            - 'sentences': List of sentences from the entire document
    """
    try:
        # Initialize the PDF reader
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Initialize result dictionary
            result = {
                'pages': [],
                'paragraphs': [],
                'sentences': []
            }
            
            # Extract text from each page
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                result['pages'].append(page_text)
                
                # Split into paragraphs (assuming paragraphs are separated by double newlines)
                paragraphs = [p.strip() for p in page_text.split('\n\n') if p.strip()]
                result['paragraphs'].extend(paragraphs)
                
                # Split into sentences (basic sentence splitting)
                sentences = re.split(r'(?<=[.!?])\s+', page_text)
                sentences = [s.strip() for s in sentences if s.strip()]
                result['sentences'].extend(sentences)
            
            return result
            
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        return {
            'pages': [],
            'paragraphs': [],
            'sentences': []
        }

# # Example usage
# if __name__ == "__main__":
#     # Example usage
#     pdf_path = "Cheshtaa Bhardwaj Test.pdf"  # Replace with your PDF file path
#     extracted_text = extract_text_from_pdf(pdf_path)
    
#     # Print the results
#     print("\nExtracted Text Structure:")
#     print(f"Number of pages: {len(extracted_text['pages'])}")
#     print(f"Number of paragraphs: {len(extracted_text['paragraphs'])}")
#     print(f"Number of sentences: {len(extracted_text['sentences'])}")
    
#     # Print first paragraph and first sentence as example
#     if extracted_text['paragraphs']:
#         print("\nFirst paragraph:")
#         print(extracted_text['paragraphs'][0])
    
#     if extracted_text['sentences']:
#         print("\nFirst sentence:")
#         print(extracted_text['sentences'][0]) 