import os
from chunknorris.parsers import PdfParser
from chunknorris.chunkers import MarkdownChunker
from chunknorris.pipelines import BasePipeline

input_folder = "Sebi"                 # your PDFs
output_folder = "chunks_individual"   # NEW folder for individual chunk files

os.makedirs(output_folder, exist_ok=True)

# Initialize parser, chunker, pipeline
parser = PdfParser(use_ocr="never")
chunker = MarkdownChunker()
pipeline = BasePipeline(parser, chunker)

for file in os.listdir(input_folder):
    if not file.endswith(".pdf"):
        continue

    filepath = os.path.join(input_folder, file)
    filename = os.path.splitext(file)[0]

    print(f"📄 Processing: {file}")

    chunks = pipeline.chunk_file(filepath)

    # Save each chunk as its own file in NEW folder
    for i, chunk in enumerate(chunks, 1):
        chunk_file = os.path.join(output_folder, f"{filename}_chunk{i}.txt")
        with open(chunk_file, "w", encoding="utf-8") as f:
            f.write(chunk.get_text())
        print(f"   ✅ Saved: {chunk_file}")

print("🎉 All PDFs processed! Each chunk is now in chunks_individual/")