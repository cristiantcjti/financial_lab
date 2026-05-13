from transformers import AutoTokenizer

class SimpleChunker:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_tokens: int = 300,
    ) -> None:
        self.max_tokens = max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    
    def create_chunks(self, text_content: str):
        paragraphs = [ paragraph.strip() for paragraph in text_content.split("\n") if paragraph.strip()]
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for paragraph in paragraphs:
            size_paragraph_tokens = len(self.tokenizer.encode(paragraph, add_special_tokens=False))
            
            if current_tokens + size_paragraph_tokens > self.max_tokens and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [paragraph]
                current_tokens = size_paragraph_tokens
            else:
                current_chunk.append(paragraph)
                current_tokens += size_paragraph_tokens
        
        if current_tokens:
            chunks.append("\n\n".join(current_chunk))
        
        return chunks