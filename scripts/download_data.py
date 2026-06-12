"""
Script para baixar o dataset de fraude em cartão de crédito do Kaggle.

Na primeira execução será solicitada autenticação no Kaggle (abre o navegador).
Nas execuções seguintes usa as credenciais em cache automaticamente.

Uso:
    uv run scripts/download_data.py
"""

import shutil
import sys
from pathlib import Path

DATASET = "mlg-ulb/creditcardfraud"
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
TARGET = RAW_DIR / "creditcard.csv"


def download():
    try:
        import kagglehub
    except ImportError:
        print("Pacote 'kagglehub' não encontrado. Instale com: uv add kagglehub")
        sys.exit(1)

    if TARGET.exists():
        print(f"Dataset já existe em {TARGET}. Pulando download.")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Baixando dataset '{DATASET}'...")
    path = kagglehub.dataset_download(DATASET)

    csv_source = next(Path(path).glob("*.csv"), None)
    if csv_source is None:
        print(f"Erro: nenhum arquivo CSV encontrado em {path}")
        sys.exit(1)

    shutil.copy(csv_source, TARGET)
    print(f"Download concluído: {TARGET}")


if __name__ == "__main__":
    download()
