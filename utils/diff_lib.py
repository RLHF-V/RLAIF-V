# %%
import re
import difflib

# SGR color constants
# rene-d 2018


class Colors:
    """ ANSI color codes """
    BLACK = "\033[0;30m"
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    BROWN = "\033[0;33m"
    BLUE = "\033[0;34m"
    PURPLE = "\033[0;35m"
    CYAN = "\033[0;36m"
    LIGHT_GRAY = "\033[0;37m"
    DARK_GRAY = "\033[1;30m"
    LIGHT_RED = "\033[1;31m"
    LIGHT_GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
    LIGHT_BLUE = "\033[1;34m"
    LIGHT_PURPLE = "\033[1;35m"
    LIGHT_CYAN = "\033[1;36m"
    LIGHT_WHITE = "\033[1;37m"
    BOLD = "\033[1m"
    FAINT = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    BLINK = "\033[5m"
    NEGATIVE = "\033[7m"
    CROSSED = "\033[9m"
    END = "\033[0m"
    # cancel SGR codes if we don't write to a terminal
    if not __import__("sys").stdout.isatty():
        for _ in dir():
            if isinstance(_, str) and _[0] != "_":
                locals()[_] = ""
    else:
        # set Windows console in VT mode
        if __import__("platform").system() == "Windows":
            kernel32 = __import__("ctypes").windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            del kernel32


def split_into_clauses(text):
    # 使用正则表达式将文本分割成子句
    # 这个正则表达式匹配句点、感叹号和问号，并且这些符号后面是空格、换行符或文本的末尾
    clauses = re.split(r'(?<=[.!?,;(.")(!")(,")(?")])\s+', text)
    return clauses


def split_into_words(text):
    # words = re.findall(r"[\w']+|[.,!?;]", text)
    words = text.split()
    return words


def show_mark_compare_words(text1, text2):
    d = difflib.Differ()
    diff = d.compare(text1.split(" "), text2.split(" "))
    return '\n'.join(diff)


def show_mark_compare_substring(text1, text2):
    d = difflib.Differ()
    diff = d.compare(split_into_clauses(text1), split_into_clauses(text2))
    return '\n'.join(diff)


def complete_modification_spans(matches, length):
    i, j = 0, matches[0][0]
    out = []
    for idx in range(0, len(matches)):
        out.append((i, j))
        out.append(matches[idx])
        if idx + 1 < len(matches):
            i, j = matches[idx][1], matches[idx + 1][0]
        else:
            i, j = matches[idx][1], length
    return out


def colorize(raw_text, color):
    return f'{color}{raw_text}{Colors.END}'

def split_mark(raw_text):
    return f'【【--{raw_text}--】】'

def color_print_diff_single(seq, diff_spans, sep=' ', color=Colors.RED, use_split=False):
    seq = list(map(str, seq))

    out = ''
    for idx, span in enumerate(diff_spans):
        text = sep.join(seq[span[0]: span[1]])
        if not text:
            continue
        if idx % 2 == 0:
            if use_split:
                out += f'{sep}{split_mark(text)}'
            else:
                out += f'{sep}{colorize(text, color)}'
        else:
            if use_split:
                out += f'{sep}{text}'
            else:
                out += f'{sep}{colorize(text, Colors.BLACK)}'
    out = out[len(sep):]
    print(f'{out}')


def get_match_info(a_seq, b_seq, min_match_size=1):
    sm = difflib.SequenceMatcher(None, a_seq, b_seq)

    mb = sm.get_matching_blocks()

    mb = [m for m in mb[:-1] if m[2] >= min_match_size] + [mb[-1]]

    a_matches = [(x[0], x[0] + x[2]) for x in mb]
    b_matches = [(x[1], x[1] + x[2]) for x in mb]
    return a_matches, b_matches


def span_not_empty(span):
    return span[0] != span[1]


def join_by_space(seq):
    return ' '.join([str(x) for x in seq])


def generate_modification_mapping_impl(a_seq, b_seq, a_spans, b_spans, do_print=False):
    assert len(a_spans) == len(b_spans)
    mod_map = {}

    if do_print:
        print(a_spans)
        print(b_spans)

    for idx, (a_span, b_span) in enumerate(zip(a_spans, b_spans)):
        if idx % 2 == 1:
            continue
        a_text = join_by_space(a_seq[a_span[0]: a_span[1]])
        b_text = join_by_space(b_seq[b_span[0]: b_span[1]])
        if do_print:
            print(f'@{colorize(a_text, Colors.RED)}@ ==> @{colorize(b_text, Colors.GREEN)}@')

        if span_not_empty(a_span) and span_not_empty(b_span):
            mod_map[a_span] = b_span

    return mod_map


def generate_modification_mapping(a_seq, b_seq, min_match_size=3, do_print=False):
    a_matches, b_matches = get_match_info(a_seq, b_seq, min_match_size=min_match_size)

    a_spans = complete_modification_spans(a_matches, len(a_seq))
    b_spans = complete_modification_spans(b_matches, len(b_seq))
    return generate_modification_mapping_impl(a_seq, b_seq, a_spans, b_spans, do_print=do_print)


def spans2ids(spans):
    ids = []
    for span in spans:
        ids += list(range(span[0], span[1]))
    return ids


def get_diff_ids(a_seq, b_seq, min_match_size=3):
    mod_map = generate_modification_mapping(a_seq, b_seq, min_match_size=min_match_size)
    a_modification_spans = list(mod_map.keys())
    b_modification_spans = list(mod_map.values())

    a_ids = sorted(set(spans2ids(a_modification_spans)))
    b_ids = sorted(set(spans2ids(b_modification_spans)))
    return a_ids, b_ids


def color_print_diff_pair(a_seq, b_seq, min_match_size=1, sep=' ', use_split=False):
    a_matches, b_matches = get_match_info(a_seq, b_seq, min_match_size=min_match_size)

    a_spans = complete_modification_spans(a_matches, len(a_seq))
    b_spans = complete_modification_spans(b_matches, len(b_seq))

    color_print_diff_single(a_seq, a_spans, sep, Colors.RED, use_split)
    color_print_diff_single(b_seq, b_spans, sep, Colors.GREEN, use_split)
