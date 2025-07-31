# utils/text_cleaner.py

import re
import html
import unicodedata

def clean_text(text: str, lower=True, normalize_unicode=True, remove_urls=True, strip=True) -> str:
    """
    Limpieza estándar del texto para análisis de sentimiento.

    Parámetros:
        text (str): texto original
        lower (bool): pasar a minúsculas
        normalize_unicode (bool): normalizar caracteres con acento
        remove_urls (bool): eliminar urls
        strip (bool): quitar espacios en extremos

    Retorna:
        str: texto limpio
    """
    if not isinstance(text, str):
        return ""

    if normalize_unicode:
        text = unicodedata.normalize("NFKD", text)
        text = "".join([c for c in text if not unicodedata.combining(c)])

    if remove_urls:
        text = re.sub(r"http\S+|www\.\S+", "", text)

    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)

    if lower:
        text = text.lower()

    if strip:
        text = text.strip()

    return text
