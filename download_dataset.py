from datasets import load_dataset
import json
import random
import ast


def format_qa(row):
    """Her bir satırı istenen formata dönüştürür."""
    # Input'tan soru ve seçenekleri ayır
    input_text = row["input"]
    question_part, options_part = input_text.split("\n{", 1)

    # Soru metnini temizle
    question = question_part.replace("Q:", "").strip()

    # Seçenekleri parse et
    options_str = "{" + options_part
    options = ast.literal_eval(options_str)

    # Cevabı parse et
    answer = row["output"].split(":")[0].strip()
    explanation = row["output"].split(":", 1)[1].strip() if ":" in row["output"] else ""

    return {
        "question": question,
        "options": options,
        "answer": answer,
        "explanation": explanation,
    }


def download_and_convert_dataset():
    print("Veri seti indiriliyor...")
    dataset = load_dataset("medalpaca/medical_meadow_medqa")

    print("Veri seti dönüştürülüyor...")
    all_data = []
    for row in dataset["train"]:
        try:
            formatted_qa = format_qa(row)
            all_data.append(formatted_qa)
        except Exception as e:
            print(f"Hata: {e}")
            print(f"Problematik veri: {row}")
            continue

    print(f"Toplam dönüştürülen veri sayısı: {len(all_data)}")

    # Veriyi karıştır
    random.seed(42)  # Tekrarlanabilirlik için
    random.shuffle(all_data)

    # Train ve validation olarak böl (90% train, 10% validation)
    split_idx = int(len(all_data) * 0.9)
    train_data = all_data[:split_idx]
    validation_data = all_data[split_idx:]

    print(f"Train veri sayısı: {len(train_data)}")
    print(f"Validation veri sayısı: {len(validation_data)}")

    print("Veri seti kaydediliyor...")

    # Train veri setini kaydet
    with open("train.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)

    # Validation veri setini kaydet
    with open("validation.json", "w", encoding="utf-8") as f:
        json.dump(validation_data, f, ensure_ascii=False, indent=4)

    print("İşlem tamamlandı!")


if __name__ == "__main__":
    download_and_convert_dataset()
