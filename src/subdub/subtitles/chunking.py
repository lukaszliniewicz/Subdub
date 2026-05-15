import logging
import srt
import re
import math
import json
import os
from typing import List, Dict
from sentence_splitter import SentenceSplitter

logger = logging.getLogger(__name__)

def create_word_blocks(words: List[Dict], char_limit: int, context_words_count: int = 6) -> List[Dict]:
    """
    Groups a list of word objects into blocks based on character limit and sentence endings,
    and includes a few subsequent words for context.
    """
    if not words:
        return []

    sentence_endings = ('.', '!', '?', '。', '！', '？', '…', ':', ';', ',')

    def is_sentence_ending(word_text: str) -> bool:
        return any(word_text.strip().endswith(ending) for ending in sentence_endings)

    all_blocks_of_words = []
    current_block = []
    current_char_count = 0

    for i, word in enumerate(words):
        word_text = word.get('word', '')
        
        if current_block and (current_char_count + len(word_text) > char_limit):
            split_at_index_in_block = -1
            for j in range(len(current_block) - 1, -1, -1):
                if is_sentence_ending(current_block[j]['word']):
                    split_at_index_in_block = j
                    break
            
            if split_at_index_in_block != -1:
                final_block = current_block[:split_at_index_in_block + 1]
                all_blocks_of_words.append(final_block)
                current_block = current_block[split_at_index_in_block + 1:]
                current_char_count = sum(len(w['word']) for w in current_block)
            else:
                all_blocks_of_words.append(current_block)
                current_block = []
                current_char_count = 0
        
        current_block.append(word)
        current_char_count += len(word_text)

    if current_block:
        all_blocks_of_words.append(current_block)
    
    final_structured_blocks = []
    total_words_processed = 0
    for block in all_blocks_of_words:
        if not block: continue
        
        start_index = total_words_processed
        end_index = start_index + len(block)
        
        context_start = end_index
        context_end = context_start + context_words_count
        context_words = words[context_start:context_end]
        
        final_structured_blocks.append({
            "words_to_process": block,
            "context_words": context_words
        })
        total_words_processed += len(block)

    logger.info(f"Created {len(final_structured_blocks)} word blocks for re-segmentation.")
    return final_structured_blocks

def create_translation_blocks(srt_content: str, char_limit: int, source_language: str) -> List[List[Dict]]:
    if source_language.lower() in ['chinese', 'japanese', 'ja', 'zh', 'zh-cn', 'zh-tw']:
        char_limit = math.floor(char_limit / 2)

    sentence_endings = {
        'default': ('.', '!', '?'),
        'japanese': ('。', '！', '？', 'か', 'ね', 'よ', 'わ'),
        'chinese': ('。', '！', '？', '…')
    }

    if source_language.lower() in ['japanese', 'ja']:
        endings = sentence_endings['japanese']
    elif source_language.lower() in ['chinese', 'zh', 'zh-cn', 'zh-tw']:
        endings = sentence_endings['chinese']
    else:
        endings = sentence_endings['default']

    subtitles = list(srt.parse(srt_content))
    blocks = []
    current_block = []
    current_char_count = 0

    def is_sentence_ending(text: str) -> bool:
        return any(text.strip().endswith(ending) for ending in endings)

    for i, subtitle in enumerate(subtitles):
        if current_block and (current_char_count + len(subtitle.content) > char_limit):
            if is_sentence_ending(current_block[-1]['text']):
                blocks.append(current_block)
                current_block = []
                current_char_count = 0
            else:
                for j in range(len(current_block) - 1, -1, -1):
                    if is_sentence_ending(current_block[j]['text']):
                        blocks.append(current_block[:j+1])
                        current_block = current_block[j+1:]
                        current_char_count = sum(len(sub['text']) for sub in current_block)
                        break
                else:
                    blocks.append(current_block)
                    current_block = []
                    current_char_count = 0

        current_block.append({
            "index": subtitle.index, 
            "text": subtitle.content,
            "start": subtitle.start.total_seconds(),
            "end": subtitle.end.total_seconds()
        })
        current_char_count += len(subtitle.content)

    if current_block:
        blocks.append(current_block)

    total_subtitles_in_blocks = sum(len(b) for b in blocks)
    logger.info(f"Created {len(blocks)} translation blocks with a total of {total_subtitles_in_blocks} subtitles.")

    return blocks

XTTS_TO_SENTENCE_SPLITTER_LANG = {
    "en": "en", "es": "es", "fr": "fr", "de": "de", "it": "it", "pt": "pt", "pl": "pl",
    "tr": "tr", "ru": "ru", "nl": "nl", "cs": "cs", "hu": "hu", "ca": "ca",
    "da": "da", "fi": "fi", "el": "el", "is": "is", "lv": "lv", "lt": "lt",
    "no": "no", "ro": "ro", "sk": "sk", "sl": "sl", "sv": "sv",
}

def _check_split_validity(text_to_split: str, split_index: int, max_len: int, min_len_part: int) -> bool:
    if split_index <= 0 or split_index >= len(text_to_split):
        return False
    
    part1 = text_to_split[:split_index].strip()
    part2 = text_to_split[split_index:].strip()

    if not part1 or not part2:
        return False
    
    return min_len_part <= len(part1) <= max_len and len(part2) >= min_len_part

def _split_further(text: str, xtts_lang_code: str, max_chars: int, min_chars: int, conjunctions_map: Dict) -> List[str]:
    text = text.strip()
    if not text: return []
    if len(text) <= max_chars:
        return [text]

    results: List[str] = []
    midpoint = len(text) // 2

    punct_sets = ['.!?', ',;:']
    best_punct_split_point = -1

    for p_set_str in punct_sets:
        min_dist_to_mid_punct = float('inf')
        current_best_for_set = -1
        for i in range(len(text) - 1, min_chars - 1, -1):
            char = text[i]
            if char in p_set_str:
                if _check_split_validity(text, i + 1, max_chars, min_chars):
                    dist = abs((i + 1) - midpoint)
                    if dist < min_dist_to_mid_punct:
                        min_dist_to_mid_punct = dist
                        current_best_for_set = i + 1
                    elif dist == min_dist_to_mid_punct and (i + 1) > current_best_for_set:
                        current_best_for_set = i + 1
        if current_best_for_set != -1:
            best_punct_split_point = current_best_for_set
            break
    
    if best_punct_split_point != -1:
        part1 = text[:best_punct_split_point].strip()
        part2 = text[best_punct_split_point:].strip()
        if part1: results.append(part1)
        if part2: results.extend(_split_further(part2, xtts_lang_code, max_chars, min_chars, conjunctions_map))
        return [r for r in results if r]

    lang_conjunctions = conjunctions_map.get(xtts_lang_code, [])
    best_conj_split_point = -1
    min_dist_to_mid_conj = float('inf')

    if lang_conjunctions:
        for conj in lang_conjunctions:
            for m in re.finditer(r'\b' + re.escape(conj) + r'\b', text, re.IGNORECASE):
                split_at_index = m.start()
                if split_at_index == 0: continue
                if _check_split_validity(text, split_at_index, max_chars, min_chars):
                    dist = abs(split_at_index - midpoint)
                    if dist < min_dist_to_mid_conj:
                        min_dist_to_mid_conj = dist
                        best_conj_split_point = split_at_index
                    elif dist == min_dist_to_mid_conj and split_at_index > best_conj_split_point:
                        best_conj_split_point = split_at_index
    
    if best_conj_split_point != -1:
        part1 = text[:best_conj_split_point].strip()
        part2 = text[best_conj_split_point:].strip()
        if part1: results.append(part1)
        if part2: results.extend(_split_further(part2, xtts_lang_code, max_chars, min_chars, conjunctions_map))
        return [r for r in results if r]

    best_word_split_point = -1
    min_dist_to_mid_word = float('inf')

    for i in range(len(text) - 1, 0, -1):
        if text[i].isspace():
            if _check_split_validity(text, i, max_chars, min_chars):
                dist = abs(i - midpoint)
                if dist < min_dist_to_mid_word:
                    min_dist_to_mid_word = dist
                    best_word_split_point = i
                elif dist == min_dist_to_mid_word and i > best_word_split_point:
                     best_word_split_point = i
    
    if best_word_split_point != -1:
        part1 = text[:best_word_split_point].strip()
        part2 = text[best_word_split_point+1:].strip()
        if part1: results.append(part1)
        if part2: results.extend(_split_further(part2, xtts_lang_code, max_chars, min_chars, conjunctions_map))
        return [r for r in results if r]

    cut_at = max_chars
    part1_final = text[:cut_at].strip()
    part2_final = text[cut_at:].strip()
    
    temp_cut_text = text[:max_chars]
    last_space = temp_cut_text.rfind(' ')

    if last_space != -1:
        temp_part1 = temp_cut_text[:last_space].strip()
        temp_part2 = text[last_space+1:].strip()
        if len(temp_part1) >= min_chars and (len(temp_part2) >= min_chars or not temp_part2):
            part1_final = temp_part1
            part2_final = temp_part2
            
    if part1_final: results.append(part1_final)
    if part2_final: results.extend(_split_further(part2_final, xtts_lang_code, max_chars, min_chars, conjunctions_map))
    return [r for r in results if r]

def _fallback_original_find_split_point(text: str, max_length: int, min_length: int, language_code: str, conjunctions_map: Dict) -> int:
    mid = len(text) // 2
    best_split_len = -1

    punct_sets_with_priority = [('.!?', 1), (',;:', 2)]

    for p_chars, priority in punct_sets_with_priority:
        current_best_len_for_set = -1
        min_dist_for_set = float('inf')
        for i in range(min_length -1, min(len(text) - min_length, max_length)):
            if text[i] in p_chars:
                part1_len = i + 1
                part2_text = text[i+1:].strip()
                if part1_len >= min_length and part1_len <= max_length and len(part2_text) >= min_length:
                    dist = abs(part1_len - mid)
                    if dist < min_dist_for_set:
                        min_dist_for_set = dist
                        current_best_len_for_set = part1_len
                    elif dist == min_dist_for_set and part1_len > current_best_len_for_set:
                        current_best_len_for_set = part1_len
        if current_best_len_for_set != -1:
            return current_best_len_for_set

    lang_conjunctions = conjunctions_map.get(language_code, [])
    min_dist_p3 = float('inf')
    best_conj_split_len = -1
    if lang_conjunctions:
        for conj in lang_conjunctions:
            for m in re.finditer(r'\b' + re.escape(conj) + r'\b', text, re.IGNORECASE):
                part1_len = m.start()
                part2_text = text[m.start():].strip()
                if part1_len >= min_length and part1_len <= max_length and len(part2_text) >= min_length:
                    dist = abs(part1_len - mid)
                    if dist < min_dist_p3:
                        min_dist_p3 = dist
                        best_conj_split_len = part1_len
                    elif dist == min_dist_p3 and part1_len > best_conj_split_len:
                        best_conj_split_len = part1_len
    if best_conj_split_len != -1:
        return best_conj_split_len

    min_dist_p4 = float('inf')
    best_word_boundary_len = -1
    for i in range(min(len(text) - 1, max_length), min_length -1 , -1):
         if text[i].isspace():
            part1_len = len(text[:i].strip())
            part2_text = text[i+1:].strip()
            if part1_len >= min_length and part1_len <= max_length and len(part2_text) >= min_length:
                dist = abs(part1_len - mid)
                if dist < min_dist_p4:
                    min_dist_p4 = dist
                    best_word_boundary_len = part1_len
                elif dist == min_dist_p4 and part1_len > best_word_boundary_len:
                    best_word_boundary_len = part1_len

    if best_word_boundary_len != -1:
        return best_word_boundary_len
    
    if len(text) > max_length:
        cut_text = text[:max_length]
        last_space = -1
        for i in range(max_length -1 , min_length -2, -1):
             if i < 0: break
             if cut_text[i].isspace():
                 if len(text[i+1:].strip()) >= min_length:
                    last_space = i
                    break
        if last_space != -1:
            return last_space
        return max_length

    return len(text)

def create_speech_blocks(
    srt_content: str,
    session_folder: str,
    video_name: str,
    target_language: str,
    min_chars: int = 10,
    max_chars: int = 160,
    merge_threshold: int = 250
) -> List[Dict]:
    
    CONJUNCTIONS = {
        "en": ["and", "but", "or", "because", "although", "so", "while", "if", "then", "that", "as", "for", "since", "until", "whether"],
        "es": ["y", "pero", "o", "porque", "aunque", "así", "mientras", "si", "entonces", "que", "como", "pues", "desde", "hasta", "si"],
        "fr": ["et", "mais", "ou", "parce que", "bien que", "donc", "pendant que", "si", "alors", "que", "comme", "car", "depuis", "jusqu'à", "si"],
        "de": ["und", "aber", "oder", "weil", "obwohl", "also", "während", "wenn", "dann", "dass", "als", "denn", "seit", "bis", "ob"],
        "it": ["e", "ma", "o", "perché", "sebbene", "quindi", "mentre", "se", "allora", "che", "come", "poiché", "da quando", "fino a", "se"],
        "pt": ["e", "mas", "ou", "porque", "embora", "então", "enquanto", "se", "logo", "que", "como", "pois", "desde", "até", "se"],
        "pl": ["i", "ale", "lub", "ponieważ", "chociaż", "więc", "podczas gdy", "jeśli", "wtedy", "że", "jak", "gdyż", "od", "aż", "czy"],
        "tr": ["ve", "ama", "veya", "çünkü", "rağmen", "bu yüzden", "iken", "eğer", "o zaman", "ki", "gibi", "zira", "-den beri", "-e kadar", "acaba"],
        "ru": ["и", "но", "или", "потому что", "хотя", "так что", "пока", "если", "тогда", "что", "как", "ибо", "с", "до", "ли"],
        "nl": ["en", "maar", "of", "omdat", "hoewel", "dus", "terwijl", "als", "dan", "dat", "zoals", "want", "sinds", "tot", "of"],
        "cs": ["a", "ale", "nebo", "protože", "ačkoli", "takže", "zatímco", "jestli", "pak", "že", "jako", "neboť", "od", "až", "zda"],
        "ar": ["و", "لكن", "أو", "لأن", "رغم أن", "لذلك", "بينما", "إذا", "ثم", "أن", "كما", "ف", "منذ", "حتى", "هل"],
        "zh-cn": ["和", "但是", "或者", "因为", "虽然", "所以", "当", "如果", "那么", "那", "作为", "由于", "自从", "直到", "是否"],
        "ja": ["そして", "しかし", "または", "なぜなら", "にもかかわらず", "だから", "間", "もし", "その時", "と", "ように", "から", "以来", "まで", "かどうか"],
        "hu": ["és", "de", "vagy", "mert", "bár", "tehát", "míg", "ha", "akkor", "hogy", "mint", "hiszen", "óta", "ameddig", "vajon"],
        "ko": ["그리고", "하지만", "또는", "왜냐하면", "비록", "그래서", "동안", "만약", "그때", "것", "처럼", "때문에", "이후", "까지", "인지"]
    }

    def get_xtts_language_code_local(lang_name: str) -> str:
        xtts_language_map = {
            "English": "en", "Spanish": "es", "French": "fr", "German": "de",
            "Italian": "it", "Portuguese": "pt", "Polish": "pl", "Turkish": "tr",
            "Russian": "ru", "Dutch": "nl", "Czech": "cs", "Arabic": "ar",
            "Chinese": "zh-cn", "Japanese": "ja", "Hungarian": "hu", "Korean": "ko",
            "en": "en", "es": "es", "fr": "fr", "de": "de", "it": "it", "pt": "pt", "pl": "pl",
            "tr": "tr", "ru": "ru", "nl": "nl", "cs": "cs", "ar": "ar", "zh-cn": "zh-cn",
            "ja": "ja", "hu": "hu", "ko": "ko", "ca": "ca", "da": "da", "fi": "fi", "el": "el",
            "is": "is", "lv": "lv", "lt": "lt", "no": "no", "ro": "ro", "sk": "sk", "sl": "sl", "sv": "sv"
        }
        return xtts_language_map.get(lang_name, lang_name if len(lang_name) == 2 else "en")

    xtts_language_code = get_xtts_language_code_local(target_language)
    
    splitter_lang_code = XTTS_TO_SENTENCE_SPLITTER_LANG.get(xtts_language_code)
    sentence_splitter_instance = None
    if splitter_lang_code:
        try:
            sentence_splitter_instance = SentenceSplitter(language=splitter_lang_code)
        except Exception as e:
            logger.warning(f"Could not initialize SentenceSplitter for {splitter_lang_code} ({xtts_language_code}): {e}. Will use fallback logic.")
            splitter_lang_code = None

    subtitles = list(srt.parse(srt_content))
    all_speech_block_parts: List[Dict] = []

    for subtitle in subtitles:
        subtitle_text = subtitle.content.strip()
        subtitle_indices = [subtitle.index]

        if not subtitle_text:
            continue

        current_sub_texts: List[str] = []

        if len(subtitle_text) >= min_chars and len(subtitle_text) <= max_chars:
            current_sub_texts.append(subtitle_text)
        elif len(subtitle_text) < min_chars:
            current_sub_texts.append(subtitle_text)
        else:
            if sentence_splitter_instance and splitter_lang_code:
                raw_sentences = sentence_splitter_instance.split(text=subtitle_text)
                for sentence in raw_sentences:
                    sentence = sentence.strip()
                    if not sentence: continue
                    if len(sentence) >= min_chars and len(sentence) <= max_chars:
                        current_sub_texts.append(sentence)
                    elif len(sentence) < min_chars:
                         current_sub_texts.append(sentence)
                    else:
                        split_parts = _split_further(sentence, xtts_language_code, max_chars, min_chars, CONJUNCTIONS)
                        current_sub_texts.extend(p for p in split_parts if p)
            else:
                remaining_text = subtitle_text
                while remaining_text:
                    if len(remaining_text) <= max_chars:
                        if remaining_text: current_sub_texts.append(remaining_text)
                        break
                    
                    split_len = _fallback_original_find_split_point(remaining_text, max_chars, min_chars, xtts_language_code, CONJUNCTIONS)
                    
                    part_to_add = remaining_text[:split_len].strip()
                    if part_to_add: current_sub_texts.append(part_to_add)
                    remaining_text = remaining_text[split_len:].strip()
        
        for text_part in current_sub_texts:
            if text_part:
                 all_speech_block_parts.append({
                    "text": text_part,
                    "subtitles": subtitle_indices 
                })
    
    final_speech_blocks: List[Dict] = []
    if not all_speech_block_parts:
        if os.path.exists(session_folder) and video_name:
            json_output_path = os.path.join(session_folder, f"{video_name}_speech_blocks.json")
            with open(json_output_path, 'w', encoding='utf-8') as json_file:
                json.dump([], json_file, ensure_ascii=False, indent=2)
        return []

    for part_data in all_speech_block_parts:
        text = part_data["text"]
        indices = part_data["subtitles"]

        if not final_speech_blocks:
            if text:
                final_speech_blocks.append({
                    "number": str(len(final_speech_blocks) + 1).zfill(4),
                    "text": text,
                    "subtitles": indices
                })
        else:
            if len(text) < min_chars and text and \
               len(final_speech_blocks[-1]["text"]) + len(text) + 1 <= max_chars:
                final_speech_blocks[-1]["text"] += " " + text
                final_speech_blocks[-1]["subtitles"] = sorted(list(set(final_speech_blocks[-1]["subtitles"] + indices)))
            elif text:
                final_speech_blocks.append({
                    "number": str(len(final_speech_blocks) + 1).zfill(4),
                    "text": text,
                    "subtitles": indices
                })
    
    for i, block in enumerate(final_speech_blocks):
        block["number"] = str(i + 1).zfill(4)

    if os.path.exists(session_folder) and video_name:
        json_output_path = os.path.join(session_folder, f"{video_name}_speech_blocks.json")
        try:
            with open(json_output_path, 'w', encoding='utf-8') as json_file:
                json.dump(final_speech_blocks, json_file, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save speech blocks JSON: {e}")
    else:
        logger.warning("Session folder or video name not available, skipping saving of speech_blocks.json")

    return final_speech_blocks
