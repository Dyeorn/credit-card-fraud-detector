"""
Script para baixar o dataset de fraude em cartão de crédito do Kaggle.

Na primeira execução será solicitada autenticação no Kaggle (abre o navegador).
Nas execuções seguintes usa as credenciais em cache automaticamente.

Uso:
    uv run scripts/download_data.py
"""

import os
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

    has_credentials = (
        (Path.home() / ".kaggle" / "kaggle.json").exists()
        or (Path.home() / ".kaggle" / "access_token").exists()
        or os.getenv("KAGGLE_USERNAME")
        or os.getenv("KAGGLE_KEY")
        or os.getenv("KAGGLE_API_TOKEN")
    )
    if not has_credentials:
        print(
            "Credenciais do Kaggle não encontradas. Configure com uma das opções:\n\n"
            "  A) Login OAuth (recomendado, apenas fora do Docker):\n"
            "       uv run -m kaggle auth login\n\n"
            "  B) Token de API (funciona no Docker):\n"
            "       export KAGGLE_API_TOKEN=<seu_token>\n\n"
            "  C) Arquivo kaggle.json (legado):\n"
            "       Baixe em https://www.kaggle.com/settings -> API\n"
            "       e salve em ~/.kaggle/kaggle.json"
        )
        sys.exit(1)

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
