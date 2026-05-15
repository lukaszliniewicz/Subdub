import os
import json
from typing import Dict

def manage_glossary(session_folder: str) -> Dict[str, str]:
    glossary_path = os.path.join(session_folder, "translation_glossary.json")
    if os.path.exists(glossary_path):
        with open(glossary_path, 'r', encoding='utf-8') as f:
            glossary = json.load(f)
    else:
        glossary = {}
    return glossary

def save_glossary(session_folder: str, glossary: Dict[str, str]) -> None:
    glossary_path = os.path.join(session_folder, "translation_glossary.json")
    with open(glossary_path, 'w', encoding='utf-8') as f:
        json.dump(glossary, f, ensure_ascii=False, indent=2)
