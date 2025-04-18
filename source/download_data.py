from datasets import load_dataset
import os

def save_wikitext_paragraphs(dataset_split="train", max_articles=200, folder="data/raw_documents"):
    os.makedirs(folder, exist_ok=True)
    dataset = load_dataset("wikitext", "wikitext-103-v1", split=dataset_split)

    buffer = []
    saved = 0

    for entry in dataset:
        line = entry["text"].strip()

        if line == "":
            if buffer:
                paragraph = " ".join(buffer)
                if paragraph.startswith("="):  # skip section titles
                    buffer = []
                    continue
                filename = f"article_{saved+1}.txt"
                with open(os.path.join(folder, filename), "w", encoding="utf-8") as f:
                    f.write(paragraph)
                print(f"✅ Saved {filename}")
                saved += 1
                buffer = []

                if saved >= max_articles:
                    break
        else:
            buffer.append(line)

if __name__ == "__main__":
    save_wikitext_paragraphs()