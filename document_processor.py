# src/document_processor.py
import os
import PyPDF2
import docx
import html2text
from typing import List, Dict, Any
from dataclasses import dataclass
import hashlib
import re

@dataclass
class DocumentChunk:
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    embeddings: List[float] = None

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False
        
    def process_document(self, file_path: str, doc_id: str) -> List[DocumentChunk]:
        """Process document based on file type"""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            return self._process_pdf(file_path, doc_id)
        elif file_ext in ['.docx', '.doc']:
            return self._process_docx(file_path, doc_id)
        elif file_ext == '.txt':
            return self._process_txt(file_path, doc_id)
        elif file_ext in ['.html', '.htm']:
            return self._process_html(file_path, doc_id)
        elif file_ext == '.md':
            return self._process_markdown(file_path, doc_id)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
    
    def _process_pdf(self, file_path: str, doc_id: str) -> List[DocumentChunk]:
        """Process PDF documents"""
        chunks = []
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                page_chunks = self._chunk_text(text, doc_id, page_num + 1)
                chunks.extend(page_chunks)
                
        return chunks
    
    def _process_docx(self, file_path: str, doc_id: str) -> List[DocumentChunk]:
        """Process DOCX documents"""
        doc = docx.Document(file_path)
        full_text = []
        
        for paragraph in doc.paragraphs:
            full_text.append(paragraph.text)
        
        text = '\n'.join(full_text)
        return self._chunk_text(text, doc_id, 1)
    
    def _process_txt(self, file_path: str, doc_id: str) -> List[DocumentChunk]:
        """Process TXT documents"""
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return self._chunk_text(text, doc_id, 1)
    
    def _process_html(self, file_path: str, doc_id: str) -> List[DocumentChunk]:
        """Process HTML documents"""
        with open(file_path, 'r', encoding='utf-8') as file:
            html_content = file.read()
        
        text = self.html_converter.handle(html_content)
        return self._chunk_text(text, doc_id, 1)
    
    def _process_markdown(self, file_path: str, doc_id: str) -> List[DocumentChunk]:
        """Process Markdown documents"""
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return self._chunk_text(text, doc_id, 1)
    
    def _chunk_text(self, text: str, doc_id: str, page_num: int) -> List[DocumentChunk]:
        """Split text into chunks with overlap"""
        chunks = []
        
        # Semantic chunking based on paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            if len(current_chunk) + len(paragraph) <= self.chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk:
                    chunk_id = self._generate_chunk_id(doc_id, page_num, len(chunks))
                    chunks.append(DocumentChunk(
                        content=current_chunk.strip(),
                        metadata={
                            'doc_id': doc_id,
                            'page_num': page_num,
                            'chunk_num': len(chunks),
                            'source': 'paragraph_chunking'
                        },
                        chunk_id=chunk_id
                    ))
                
                # If single paragraph is larger than chunk size, split it
                if len(paragraph) > self.chunk_size:
                    sentence_chunks = self._split_large_paragraph(paragraph)
                    for sentence_chunk in sentence_chunks:
                        chunk_id = self._generate_chunk_id(doc_id, page_num, len(chunks))
                        chunks.append(DocumentChunk(
                            content=sentence_chunk,
                            metadata={
                                'doc_id': doc_id,
                                'page_num': page_num,
                                'chunk_num': len(chunks),
                                'source': 'sentence_chunking'
                            },
                            chunk_id=chunk_id
                        ))
                    current_chunk = ""
                else:
                    current_chunk = paragraph + "\n\n"
        
        # Add the last chunk
        if current_chunk.strip():
            chunk_id = self._generate_chunk_id(doc_id, page_num, len(chunks))
            chunks.append(DocumentChunk(
                content=current_chunk.strip(),
                metadata={
                    'doc_id': doc_id,
                    'page_num': page_num,
                    'chunk_num': len(chunks),
                    'source': 'paragraph_chunking'
                },
                chunk_id=chunk_id
            ))
        
        return chunks
    
    def _split_large_paragraph(self, paragraph: str) -> List[str]:
        """Split large paragraphs using sliding window approach"""
        sentences = re.split(r'[.!?]+', paragraph)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks
    
    def _generate_chunk_id(self, doc_id: str, page_num: int, chunk_num: int) -> str:
        """Generate unique chunk ID"""
        content = f"{doc_id}_{page_num}_{chunk_num}"
        return hashlib.md5(content.encode()).hexdigest()