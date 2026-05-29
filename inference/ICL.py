"""Task prompt templates used by the data collator.

Only the two prompt constants required by ``utils/data/data_collator.py`` are
shipped here. The full in-context-learning harness from the research repo is
not part of this release.
"""

Constrained_PROMPT = (
    "We will give you several examples and you should follow them to accomplish "
    "the task.\n Examples:\n"
)

TASK_PROMT = {
    "FOMC": "What is the monetary policy stance for the following text? A. dovish, B. hawkish, C. neutral. Choose one from A, B and C.\n",
    "C-STANCE": "判断以下文本对指定对象的态度，选择一项：A.支持，B.反对，C.中立。输出A，B或者C。\n",
    "ScienceQA": "Choose an answer for the following question and give your reasons.\n\n",
    "NumGLUE-cm": "Solve the following math problem.\n",
    "NumGLUE-ds": "Solve the following math problem.\n",
    "MeetingBank": "Write a summary of the following meeting transcripts.\n",
    "Py150": "Continue writing the code.\n",
    "20Minuten": "Provide a simplified version of the following paragraph in German.\n\n",
}
