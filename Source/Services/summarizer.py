"""
Content Summarizer - Summarization system for videos and documents
Uses Azure OpenAI language model to create customized summaries
"""

import os
import asyncio
import traceback
from typing import Dict, Optional
from openai import AsyncAzureOpenAI
from Config.config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_CHAT_COMPLETION_MODEL,
    CONTAINER_NAME
)
from Source.Services.blob_manager import BlobManager
from Config.logging_config import setup_logging

logger = setup_logging()


class ContentSummarizer:
    """
    Content summarization system - videos and documents
    """

    def __init__(self):
        """
        Initialize summarization system
        """
        self.model_name = AZURE_OPENAI_CHAT_COMPLETION_MODEL

        # Create async OpenAI client
        self.openai_client = AsyncAzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )

        # Create BlobManager for accessing files in blob storage
        self.blob_manager = BlobManager()
        logger.info(f"ContentSummarizer initialized with model: {self.model_name}")

    def build_base_prompt(self, subject_name: Optional[str], subject_type: Optional[str],
                          input_type: str = "file") -> str:
        """
        Returns a Hebrew, learning-oriented summarization prompt tailored by subject_type and input_type.

        Args:
            subject_name: Name of the subject
            subject_type: "מתמטי" | "הומני" | None
            input_type: "video" | "file" (generic course file)

        Returns:
            Formatted prompt string
        """

        if input_type == "video":
            input_line = "הקלט הוא תמלול של הרצאת וידאו. "
        else:
            input_line = "הקלט הוא קובץ קורס שעשוי להיות הרצאה, תרגול, שיעורי בית או כל חומר שהמרצה העלה לסטודנטים. "

        # Math-focused
        if subject_type == "מתמטי":
            if subject_name:
                return (
                    f"אתה מומחה לסיכום תכני קורס אקדמיים במקצוע {subject_name} (מסוג מתמטי). "
                    f"{input_line}"
                    "מטרת הסיכום אינה לקיצור הטקסט, אלא יצירת סיכום לימודי שמאפשר ללמוד את החומר בצורה מסודרת ומאורגנת, "
                    "ללא חזרות מיותרות, ותוך הכללה של כל התוכן הרלוונטי. "
                    "התאם במיוחד לתחומים מתמטיים: כלול הגדרות מדויקות, סימון ונוטציה עקביים, משפטים ועקרונות, "
                    "סקיצות הוכחה/אינטואיציה להוכחות, אלגוריתמים (במידת הצורך) בפסאודו־קוד קריא, ועבודה עם נוסחאות (LaTeX כאשר מתאים). "
                    "שלב דוגמאות פתורות צעד־אחר־צעד, הדגמות של טעויות נפוצות ותובנות מפתח. "
                    "ארגן באופן היררכי וברור, שמור על המינוח המקורי ככל האפשר, וכתוב בעברית ברורה וקוהרנטית.\n\n"
                    "כתוב סיכום מפורט ככל שנדרש — גם אם הוא ארוך מאוד — כך שהסטודנט יוכל ללמוד רק מהסיכום בלי לצפות בהרצאה.\n\n"
                    "מבנה הפלט:\n"
                    "1. **רשימת נושאים עיקריים** — נקודות קצרות שמסכמות את התוכן.\n"
                    f"2. **סיכום מפורט של ה{'שיעור' if input_type == 'video' else 'קובץ'}** — כולל הסברים, דוגמאות והערות של המרצה במידה ויש, כתוב בשפה ברורה ונגישה.\n"
                    "3. **המלצות ללמידה והעמקה** — הצע דרכי פעולה לחזרה, חיזוק ותרגול.\n\n"
                    "זכור:\n"
                    "- כתוב בטון מסביר ומלווה, כאילו אתה מחנך שמנגיש את החומר.\n"
                    f"- סדר את הכל בצורה שתשקף את הזרימה המקורית של ה{'הרצאה' if input_type == 'video' else 'קובץ'}."
                )
            else:
                return (
                    "אתה מומחה לסיכום תכני קורס אקדמיים בתחומים מתמטיים. "
                    f"{input_line}"
                    "מטרת הסיכום אינה לקיצור הטקסט, אלא יצירת סיכום לימודי שמאפשר ללמוד את החומר בצורה מסודרת ומאורגנת, "
                    "ללא חזרות מיותרות, ותוך הכללה של כל התוכן הרלוונטי. "
                    "התאם במיוחד לתחומים מתמטיים: כלול הגדרות מדויקות, סימון ונוטציה עקביים, משפטים ועקרונות, "
                    "סקיצות הוכחה/אינטואיציה להוכחות, אלגוריתמים (במידת הצורך) בפסאודו־קוד קריא, ועבודה עם נוסחאות (LaTeX כאשר מתאים). "
                    "שלב דוגמאות פתורות צעד־אחר־צעד, הדגמות של טעויות נפוצות ותובנות מפתח. "
                    "ארגן באופן היררכי וברור, שמור על המינוח המקורי ככל האפשר, וכתוב בעברית ברורה וקוהרנטית.\n\n"
                    "כתוב סיכום מפורט ככל שנדרש — גם אם הוא ארוך מאוד — כך שהסטודנט יוכל ללמוד רק מהסיכום בלי לצפות בהרצאה.\n\n"
                    "מבנה הפלט:\n"
                    "1. **רשימת נושאים עיקריים** — נקודות קצרות שמסכמות את התוכן.\n"
                    f"2. **סיכום מפורט של ה{'שיעור' if input_type == 'video' else 'קובץ'}** — כולל הסברים, דוגמאות והערות של המרצה במידה ויש, כתוב בשפה ברורה ונגישה.\n"
                    "3. **המלצות ללמידה והעמקה** — הצע דרכי פעולה לחזרה, חיזוק ותרגול.\n\n"
                    "זכור:\n"
                    "- כתוב בטון מסביר ומלווה, כאילו אתה מחנך שמנגיש את החומר.\n"
                    f"- סדר את הכל בצורה שתשקף את הזרימה המקורית של ה{'הרצאה' if input_type == 'video' else 'קובץ'}."
                )

        # Humanities-focused
        if subject_type == "הומני":
            if subject_name:
                return (
                    f"אתה מומחה לסיכום תכני קורס אקדמיים במקצוע {subject_name} (מסוג הומני). "
                    f"{input_line}"
                    "המטרה אינה לקצר, אלא לבנות סיכום לימודי שמאפשר ללמוד את החומר באופן מסודר ומאורגן, "
                    "ללא חזרות מיותרות, ותוך הכללה של כל התוכן הרלוונטי. "
                    "הדגש מושגים מרכזיים, הקשרים והיסטוריה, עמדות/אסכולות, טענות ונימוקים, דוגמאות ומקרי־מבחן, "
                    "וציטוטים קצרים עם ייחוס (אם רלוונטי). כתוב בעברית ברורה וקוהרנטית.\n\n"
                    "כתוב סיכום מפורט ככל שנדרש — גם אם הוא ארוך מאוד — כך שהסטודנט יוכל ללמוד רק מהסיכום בלי לצפות בהרצאה.\n\n"
                    "מבנה הפלט:\n"
                    "1. **רשימת נושאים עיקריים** — נקודות קצרות שמסכמות את התוכן.\n"
                    f"2. **סיכום מפורט של ה{'שיעור' if input_type == 'video' else 'קובץ'}** — כולל הסברים, דוגמאות והערות של המרצה במידה ויש, כתוב בשפה ברורה ונגישה.\n"
                    "3. **המלצות ללמידה והעמקה** — הצע דרכי פעולה לחזרה, חיזוק ותרגול.\n\n"
                    "זכור:\n"
                    "- כתוב בטון מסביר ומלווה, כאילו אתה מחנך שמנגיש את החומר.\n"
                    f"- סדר את הכל בצורה שתשקף את הזרימה המקורית של ה{'הרצאה' if input_type == 'video' else 'קובץ'}."
                )
            else:
                return (
                    "אתה מומחה לסיכום תכני קורס אקדמיים בתחומים הומניים. "
                    f"{input_line}"
                    "המטרה אינה לקצר, אלא לבנות סיכום לימודי שמאפשר ללמוד את החומר באופן מסודר ומאורגן, "
                    "ללא חזרות מיותרות, ותוך הכללה של כל התוכן הרלוונטי. "
                    "הדגש מושגים מרכזיים, הקשרים והיסטוריה, עמדות/אסכולות, טענות ונימוקים, דוגמאות ומקרי־מבחן, "
                    "וציטוטים קצרים עם ייחוס (אם רלוונטי). כתוב בעברית ברורה וקוהרנטית.\n\n"
                    "כתוב סיכום מפורט ככל שנדרש — גם אם הוא ארוך מאוד — כך שהסטודנט יוכל ללמוד רק מהסיכום בלי לצפות בהרצאה.\n\n"
                    "מבנה הפלט:\n"
                    "1. **רשימת נושאים עיקריים** — נקודות קצרות שמסכמות את התוכן.\n"
                    f"2. **סיכום מפורט של ה{'שיעור' if input_type == 'video' else 'קובץ'}** — כולל הסברים, דוגמאות והערות של המרצה במידה ויש, כתוב בשפה ברורה ונגישה.\n"
                    "3. **המלצות ללמידה והעמקה** — הצע דרכי פעולה לחזרה, חיזוק ותרגול.\n\n"
                    "זכור:\n"
                    "- כתוב בטון מסביר ומלווה, כאילו אתה מחנך שמנגיש את החומר.\n"
                    f"- סדר את הכל בצורה שתשקף את הזרימה המקורית של ה{'הרצאה' if input_type == 'video' else 'קובץ'}."
                )

        # Generic fallback
        return (
            "אתה מומחה לסיכום תכני קורס אקדמיים. "
            f"{input_line}"
            "המטרה אינה לקצר, אלא לבנות סיכום לימודי שמאפשר ללמוד את החומר באופן מסודר ומאורגן, "
            "ללא חזרות מיותרות, ותוך הכללה של כל התוכן הרלוונטי. "
            "ארגן את הסיכום באופן היררכי וברור; כלול מושגים מרכזיים, דוגמאות והסברים אינטואיטיביים; "
            "כתוב בעברית ברורה וקוהרנטית.\n\n"
            "כתוב סיכום מפורט ככל שנדרש — גם אם הוא ארוך מאוד — כך שהסטודנט יוכל ללמוד רק מהסיכום בלי לצפות בהרצאה.\n\n"
            "מבנה הפלט:\n"
            "1. **רשימת נושאים עיקריים** — נקודות קצרות שמסכמות את התוכן.\n"
            f"2. **סיכום מפורט של ה{'שיעור' if input_type == 'video' else 'קובץ'}** — כולל הסברים, דוגמאות והערות של המרצה במידה ויש, כתוב בשפה ברורה ונגישה.\n"
            "3. **המלצות ללמידה והעמקה** — הצע דרכי פעולה לחזרה, חיזוק ותרגול.\n\n"
            "זכור:\n"
            "- כתוב בטון מסביר ומלווה, כאילו אתה מחנך שמנגיש את החומר.\n"
            f"- סדר את הכל בצורה שתשקף את הזרימה המקורית של ה{'הרצאה' if input_type == 'video' else 'קובץ'}."
        )

    def _get_section_summary_prompt(self, subject_name: str = None, subject_type: str = None, previous_summary: str = None) -> str:
        """Prepare prompt for complete Section summarization"""

        # Build subject context
        subject_context = ""
        if subject_name and subject_type == "מתמטי":
            subject_context = (
                f"אתה מומחה לסיכום חומרי לימוד אקדמיים במקצוע {subject_name} (מסוג מתמטי). "
                "התאם במיוחד לתחומים מתמטיים: כלול הגדרות מדויקות, סימון ונוטציה עקביים, משפטים ועקרונות, "
                "סקיצות הוכחה/אינטואיציה להוכחות, אלגוריתמים (במידת הצורך) בפסאודו־קוד קריא, ועבודה עם נוסחאות (LaTeX כאשר מתאים). "
                "שלב דוגמאות פתורות צעד־אחר־צעד, הדגמות של טעויות נפוצות ותובנות מפתח. "
            )
        elif subject_name and subject_type == "הומני":
            subject_context = (
                f"אתה מומחה לסיכום חומרי לימוד אקדמיים במקצוע {subject_name} (מסוג הומני). "
                "הדגש מושגים מרכזיים, הקשרים והיסטוריה, עמדות/אסכולות, טענות ונימוקים, דוגמאות ומקרי־מבחן, "
                "וציטוטים קצרים עם ייחוס (אם רלוונטי). "
            )
        elif subject_type == "מתמטי":
            subject_context = (
                "אתה מומחה לסיכום חומרי לימוד אקדמיים בתחומים מתמטיים. "
                "התאם במיוחד לתחומים מתמטיים: כלול הגדרות מדויקות, סימון ונוטציה עקביים, משפטים ועקרונות, "
                "סקיצות הוכחה/אינטואיציה להוכחות, אלגוריתמים (במידת הצורך) בפסאודו־קוד קריא, ועבודה עם נוסחאות (LaTeX כאשר מתאים). "
                "שלב דוגמאות פתורות צעד־אחר־צעד, הדגמות של טעויות נפוצות ותובנות מפתח. "
            )
        elif subject_type == "הומני":
            subject_context = (
                "אתה מומחה לסיכום חומרי לימוד אקדמיים בתחומים הומניים. "
                "הדגש מושגים מרכזיים, הקשרים והיסטוריה, עמדות/אסכולות, טענות ונימוקים, דוגמאות ומקרי־מבחן, "
                "וציטוטים קצרים עם ייחוס (אם רלוונטי). "
            )
        else:
            subject_context = "אתה מומחה לסיכום חומרי לימוד אקדמיים. "

        # Build previous summary context
        previous_context = ""
        if previous_summary:
            previous_context = f"""
**חשוב: יש לך גם סיכום מהסקשיין הקודם לקונטקסט:**

{previous_summary}

**הוראות לשימוש בסיכום הקודם:**
- השתמש בסיכום הקודם כרקע וקונטקסט להבנת החומר הנוכחי
- אם המרצה מתייחס למושגים או נושאים שהוסברו בסקשיין הקודם, תוכל לתת הקשר מתאים
- אל תחזור על החומר מהסקשיין הקודם - רק תשתמש בו להבנה וקונטקסט
- הסיכום שלך צריך להתמקד בחומר החדש מהסקשיין הנוכחי
- אם יש קשרים או המשכיות לחומר הקודם, ציין זאת בקצרה

"""

        return f"""{subject_context}קיבלת אוסף של סיכומים כתובים (Markdown) מתוך Section שלם בקורס אוניברסיטאי.
כל סיכום מייצג שיעור, מסמך או תרגול שנלמדו באותו Section.
המטרה שלך היא לאחד את כל הסיכומים לכדי סיכום־על **מפורט**, מקיף ופדגוגי, שמציג את התמונה הכוללת של ה-Section.
אל תחסוך בפרטים — כלול הגדרות, דוגמאות, הסברים והערות חשובות שהופיעו בקבצים.

{previous_context}זכור: המטרה **אינה לקצר** את החומר אלא לארגן אותו מחדש, להרחיב ולהסביר כך שהסטודנט יוכל ללמוד את כל החומר מתוך הסיכום הסופי **ללא תלות בחומרים המקוריים**.

המטרה שלך:
- ליצור סיכום מקיף של כל ה-Section שמכסה את כל החומרים שקיבלת.
- לזהות קשרים ונושאים משותפים בין הקבצים השונים.
- לסדר את החומר בצורה לוגית ומובנת.
- ליצור מבט כולל על כל הנושאים שנלמדו ב-Section.

המשימה שלך:
- פתח את הסיכום במשפט או שניים שמציגים בקצרה מה נלמד בסקשן ומה המטרה שלו.
- עבור על כל הקבצים וזהה את הנושאים העיקריים.
- מצא קשרים והמשכיות בין הנושאים השונים.
- סדר את החומר בצורה הגיונית — מהבסיסי למתקדם או לפי רצף הלמידה.
- הדגש נקודות חשובות, מושגי מפתח ודגשים שחוזרים על עצמם.

מבנה הפלט:
1. **פתיח קצר** — משפט או שניים שמסבירים מה נלמד ומה מטרת הסקשן.
2. **סקירה כללית של ה-Section** — רשימה מסודרת של הנושאים המרכזיים.
3. **סיכום מפורט לפי נושאים** — חלוקה לוגית של החומר עם הסברים מקיפים, דוגמאות והבהרות.
4. **נקודות מפתח והמלצות ללמידה** — דגשים חשובים לזכירה ודרכי פעולה לחזרה ותרגול.

זכור:
- שמור על מבנה מסודר והגיוני שמקל על הבנה.
- כתוב בצורה ברורה, נגישה ומלווה — כאילו אתה מדריך את הסטודנט שלב אחר שלב.
- אל תדלג על פרטים חשובים — המטרה היא סיכום שלם ומקיף.

סיכומי כל הקבצים:
"""

    def _get_course_summary_prompt(self, subject_name: str = None, subject_type: str = None) -> str:
        """Prepare prompt for reorganizing complete course content"""

        # Build subject context
        subject_context = ""
        if subject_name and subject_type == "מתמטי":
            subject_context = (
                f"אתה מומחה לארגון והנגשה של חומרי לימוד אקדמיים במקצוע {subject_name} (מסוג מתמטי). "
                "התאם במיוחד לתחומים מתמטיים: כלול הגדרות מדויקות, סימון ונוטציה עקביים, משפטים ועקרונות, "
                "סקיצות הוכחה/אינטואיציה להוכחות, אלגוריתמים (במידת הצורך) בפסאודו־קוד קריא, ועבודה עם נוסחאות (LaTeX כאשר מתאים). "
                "שלב דוגמאות פתורות צעד־אחר־צעד, הדגמות של טעויות נפוצות ותובנות מפתח. "
            )
        elif subject_name and subject_type == "הומני":
            subject_context = (
                f"אתה מומחה לארגון והנגשה של חומרי לימוד אקדמיים במקצוע {subject_name} (מסוג הומני). "
                "הדגש מושגים מרכזיים, הקשרים והיסטוריה, עמדות/אסכולות, טענות ונימוקים, דוגמאות ומקרי־מבחן, "
                "וציטוטים קצרים עם ייחוס (אם רלוונטי). "
            )
        elif subject_type == "מתמטי":
            subject_context = (
                "אתה מומחה לארגון והנגשה של חומרי לימוד אקדמיים בתחומים מתמטיים. "
                "התאם במיוחד לתחומים מתמטיים: כלול הגדרות מדויקות, סימון ונוטציה עקביים, משפטים ועקרונות, "
                "סקיצות הוכחה/אינטואיציה להוכחות, אלגוריתמים (במידת הצורך) בפסאודו־קוד קריא, ועבודה עם נוסחאות (LaTeX כאשר מתאים). "
                "שלב דוגמאות פתורות צעד־אחר־צעד, הדגמות של טעויות נפוצות ותובנות מפתח. "
            )
        elif subject_type == "הומני":
            subject_context = (
                "אתה מומחה לארגון והנגשה של חומרי לימוד אקדמיים בתחומים הומניים. "
                "הדגש מושגים מרכזיים, הקשרים והיסטוריה, עמדות/אסכולות, טענות ונימוקים, דוגמאות ומקרי־מבחן, "
                "וציטוטים קצרים עם ייחוס (אם רלוונטי). "
            )
        else:
            subject_context = "אתה מומחה לארגון והנגשה של חומרי לימוד אקדמיים. "

        return f"""{subject_context}קיבלת אוסף של סיכומי Section מתוך קורס אוניברסיטאי שלם.
    כל סיכום Section מייצג חלק משמעותי מהחומר שכבר עבר עיבוד מפורט. כעת תפקידך לשלב, לארגן ולהציג מחדש את התוכן הקיים בצורה **מלאה**, **ברורה** ו**פדגוגית** — כך שסטודנט יוכל ללמוד את כל חומר הקורס מתוך תוצר אחד כולל.

    שים לב: המשימה **אינה לקצר** את החומר או להשמיט פרטים, אלא לבנות מבנה כולל, ברור ומקושר של כל תוכן הקורס.
    עליך **לשלב באופן פעיל דוגמאות, הסברים, הגדרות והערות** — אלה אינם תוספות, אלא חלק מרכזי להבנה.

    המטרה והמשימה שלך:
    - ליצור הצגה חינוכית של הקורס כולו, המבוססת על כלל ה-Sections שסופקו.
    - לזהות את המבנה הלוגי וההתפתחות הפדגוגית של הקורס.
    - לארגן את החומר באופן שמדגיש התקדמות מהבסיס למתקדם וקשרים בין נושאים.
    - **שמור על עומק, הסבר ודוגמה רלוונטית** — אל תדלג על פרטים התורמים ללמידה.
    - הדגש מושגים חוזרים, הרחבות והכללות שנבנו לאורך הקורס.

    מבנה הפלט:
    1. **פתיח קצר** — מטרות-על של הקורס וסקירה תמציתית של תחומי התוכן.
    2. **סקירה כללית של נושאי קורס** — רשימה מסודרת של הנושאים המרכזיים.
    3. **סיכום מפורט לפי נושאים** — חלוקה לוגית של החומר עם הסברים מקיפים, דוגמאות והבהרות.
    4. **נקודות מפתח והמלצות ללמידה** — דגשים חשובים לזכירה ודרכי פעולה לחזרה ותרגול.

    הצגת הקורס:
    """

    async def parse_video_md_file_from_blob(self, blob_path: str) -> Dict:
        """
        Parse video.md file from blob storage to extract transcript

        Args:
            blob_path: Path to video.md file in blob storage

        Returns:
            Dictionary with transcript content
        """
        logger.info(f"Parsing video MD file from blob: {blob_path}")

        file_bytes = await self.blob_manager.download_to_memory(blob_path)
        if not file_bytes:
            raise FileNotFoundError(f"File not found in blob: {blob_path}")

        content = file_bytes.decode('utf-8')
        full_transcript = None

        lines = content.split('\n')
        current_section = None
        section_content = []

        for line in lines:
            line_stripped = line.strip()

            if line_stripped in ["## Full Transcript", "## טרנסקריפט מלא"]:
                current_section = "full_transcript"
                section_content = []
            elif line_stripped.startswith("## ") and current_section == "full_transcript":
                if section_content:
                    full_transcript = '\n'.join(section_content).strip()
                break
            else:
                if current_section:
                    section_content.append(line)

        if current_section and section_content:
            full_transcript = '\n'.join(section_content).strip()

        return {
            "full_transcript": full_transcript,
            "original_content": content
        }

    async def summarize_content(self, content: str, content_type: str = "document", subject_name: str = None,
                                subject_type: str = None, existing_summary: str = None) -> str:
        """
        Create summary for content

        Args:
            content: Content to summarize (MD text)
            content_type: Content type - "video" or "document"
            subject_name: Subject name
            subject_type: Subject type (for video only)
            existing_summary: Existing summary (for video only)

        Returns:
            Generated summary
        """
        logger.info(f"Creating summary for {content_type} content...")
        logger.info(f"Content length: {len(content)} characters")

        try:
            # Use the new build_base_prompt function
            logger.info(f"Subject name: {subject_name}")
            logger.info(f"Subject type: {subject_type}")

            if content_type.lower() == "video":
                system_prompt = self.build_base_prompt(
                    subject_name=subject_name,
                    subject_type=subject_type,
                    input_type="video"
                )
            else:
                system_prompt = self.build_base_prompt(
                    subject_name=subject_name,
                    subject_type=subject_type,
                    input_type="file"
                )

            # Prepare messages
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": content
                }
            ]

            # Call language model
            logger.info(f"Calling {self.model_name} for summarization...")
            response = await self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.3,  # Stability in summarization
                top_p=0.7
            )

            summary = response.choices[0].message.content

            logger.info(f"Summary created successfully!")
            logger.info(f"Summary length: {len(summary)} characters")

            return summary

        except Exception as e:
            logger.info(f"Error creating summary: {e}")
            return f"Error creating summary: {str(e)}"

    def _detect_content_type_from_path(self, blob_path: str) -> str:
        """
        Identify content type by file path
        Returns 'video' if path contains 'Videos_md' or 'document' if contains 'Docs_md'
        """
        logger.info(f"blob_path: {blob_path}")
        if "videos_md" in blob_path.lower():
            return "video"
        elif "docs_md" in blob_path.lower():
            return "document"
        else:
            # Default - try to identify by extension
            if blob_path.lower().endswith('.md'):
                return "document"  # Default for documents
            return "unknown"

    def _extract_section_from_path(self, blob_path: str) -> str:
        """
        Extract section name from blob path
        Example: "Section1/Processed-data/Videos_md/file.md" -> "Section1"
        """
        path_parts = blob_path.split('/')
        for part in path_parts:
            if part.lower().startswith('section'):
                return part
        return "general"  # Default if no section found

    async def summarize_md_file(self, blob_path: str, subject_name: str = None, subject_type: str = None) -> str | None:
        """
        Summarize MD file from blob with automatic content type detection and save to blob

        Args:
            blob_path: Path to MD file in blob
            subject_name: Subject name for context
            subject_type: Subject type for context

        Returns:
            Summary path in blob or None if failed
        """
        logger.info(f"Processing MD file from blob: {blob_path}")

        try:
            # Identify content type from path
            content_type = self._detect_content_type_from_path(blob_path)
            logger.info(f"Identified as type: {content_type}")

            if content_type == "unknown":
                logger.info(f"Cannot identify file type for: {blob_path}")
                return None

            # Download file directly to memory from blob
            file_bytes = await self.blob_manager.download_to_memory(blob_path)
            if not file_bytes:
                logger.info(f"Failed to download file from blob: {blob_path}")
                return None

            # Convert to text
            content = file_bytes.decode('utf-8')
            temp_file_path = None  # No temp file needed

            try:
                # If it's a video file - use advanced parsing
                if content_type == "video":
                    logger.info("Video file detected - using enhanced parsing and summarization")

                    # Parse file content directly from blob
                    parsed_data = await self.parse_video_md_file_from_blob(blob_path)

                    # Check that transcript exists
                    if not parsed_data.get("full_transcript"):
                        logger.info(f"No transcript found in video file")
                        return None

                    # Create summary with parsed parameters
                    summary = await self.summarize_content(
                        content=parsed_data["full_transcript"],
                        content_type="video",
                        subject_name=subject_name,
                        subject_type=subject_type,
                        existing_summary=parsed_data.get("existing_summary")
                    )

                # If it's a regular document - standard handling
                else:
                    logger.info("Document file - using standard processing")

                    if not content.strip():
                        logger.info(f"File is empty")
                        return None

                    # Create summary
                    summary = await self.summarize_content(
                        content=content,
                        content_type=content_type,
                        subject_name=subject_name,
                        subject_type=subject_type
                    )

                # Check that summary was created successfully
                if not summary or summary.startswith("Error"):
                    logger.info(f"Failed to create summary")
                    return None

                # Save summary to blob
                blob_summary_path = await self._save_summary_to_blob(summary, blob_path)
                if blob_summary_path:
                    logger.info(f"Summary saved to blob: {blob_summary_path}")
                    return blob_summary_path
                else:
                    logger.info(f"Failed to save summary to blob")
                    return None

            finally:
                # Delete temporary file (only if it was created)
                if temp_file_path and os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

        except Exception as e:
            logger.info(f"Error processing file: {str(e)}")
            return None

    async def _save_summary_to_blob(self, summary: str, original_blob_path: str) -> str:
        """
        Save summary to blob in structure CourseID/SectionID/file_summaries/FileID.md

        Args:
            summary: Summary to save
            original_blob_path: Original file path in blob (e.g., "CS101/Section1/Docs_md/1.md")

        Returns:
            Summary path in blob or None if failed
        """
        try:
            # Parse original path
            # Example: "CS101/Section1/Docs_md/1.md" -> ["CS101", "Section1", "Docs_md", "1.md"]
            path_parts = original_blob_path.split('/')

            if len(path_parts) < 4:
                logger.info(f"Invalid path: {original_blob_path}")
                return None

            course_id = path_parts[0]  # CS101
            section_id = path_parts[1]  # Section1
            # path_parts[2] is Docs_md or Videos_md
            filename = path_parts[3]  # 1.md

            # Extract filename without extension
            base_name = os.path.splitext(filename)[0]  # 1

            # Add title to summary
            title = f"# סיכום קובץ {base_name}\n\n"
            summary_with_title = title + summary

            # Create new summary path
            summary_blob_path = f"{course_id}/{section_id}/file_summaries/{base_name}.md"

            logger.info(f"Saving summary to blob: {summary_blob_path}")

            # Save to blob
            success = await self.blob_manager.upload_text_to_blob(
                text_content=summary_with_title,
                blob_name=summary_blob_path,
                container=CONTAINER_NAME
            )

            if success:
                return summary_blob_path
            else:
                logger.info(f"Failed to save summary to blob")
                return None

        except Exception as e:
            logger.info(f"Error saving summary to blob: {str(e)}")
            return None

    async def summarize_section_from_blob(self, full_blob_path: str, subject_name: str = None,
                                          subject_type: str = None, previous_summary_path: str = None) -> str | None:
        """
        Summarize complete section from all summary files in blob storage
        Args:
            full_blob_path: Path to file_summaries folder (e.g., "CS101/Section1/file_summaries")
            subject_name: Subject name for context
            subject_type: Subject type for context
        Returns:
            Summary path in blob or None if failed
        """

        try:
            # Parse path: "CS101/Section1/file_summaries" -> ["CS101", "Section1", "file_summaries"]
            path_parts = full_blob_path.split('/')

            if len(path_parts) < 3:
                logger.info(f"Invalid path: {full_blob_path}. Should be in format: CourseID/SectionID/file_summaries")
                return None

            course_id = path_parts[0]  # CS101
            section_id = path_parts[1]  # Section1
            # path_parts[2] should be file_summaries

            logger.info(f"CourseID: {course_id}")
            logger.info(f"SectionID: {section_id}")
            logger.info(f"file_summaries path: {full_blob_path}")

            # Create BlobManager with default container
            blob_manager = BlobManager()

            # Get list of all files in container
            all_files = await blob_manager.list_files()

            # Filter files in specific path
            section_files = [f for f in all_files if f.startswith(full_blob_path + "/") and f.endswith(".md")]

            if not section_files:
                logger.info(f"No summary files found in {full_blob_path}")
                return None

            logger.info(f"Found {len(section_files)} summary files in {full_blob_path}:")
            for file in section_files:
                logger.info(f"  - {file}")

            # Download and read all files directly to memory
            all_content = ""
            successful_files = []

            for file_path in section_files:
                logger.info(f"\n Downloading file to memory: {file_path}")

                try:
                    # Download file directly to memory
                    file_bytes = await blob_manager.download_to_memory(file_path)

                    if file_bytes:
                        # Convert to text
                        file_content = file_bytes.decode('utf-8')

                        if file_content.strip():
                            all_content += f"\n\n{'=' * 50}\n"
                            all_content += f"קובץ: {os.path.basename(file_path)}\n"
                            all_content += f"{'=' * 50}\n\n"
                            all_content += file_content
                            successful_files.append(file_path)
                            logger.info(f" File read successfully: {len(file_content)} characters")
                        else:
                            logger.info(f"Empty file: {file_path}")
                    else:
                        logger.info(f"Failed to download file: {file_path}")

                except Exception as e:
                    logger.info(f"Error processing file {file_path}: {e}")
                    continue

            if not successful_files:
                logger.info(f"Could not read any files from {full_blob_path}")
                return None

            logger.info(f"\n Total working with {len(successful_files)} files")
            logger.info(f"Total content length: {len(all_content)} characters")

            # Handle previous summary path if provided
            previous_summary = None
            if previous_summary_path:
                try:
                    logger.info(f"Trying to read previous section summary: {previous_summary_path}")
                    previous_file_bytes = await blob_manager.download_to_memory(previous_summary_path)
                    if previous_file_bytes:
                        previous_summary = previous_file_bytes.decode('utf-8')
                        logger.info(f"Successfully loaded previous summary: {len(previous_summary)} characters")
                    else:
                        logger.warning(f"Could not download previous summary from: {previous_summary_path}")
                except Exception as e:
                    logger.warning(f"Error loading previous summary from {previous_summary_path}: {e}")

            # Create summary
            logger.info(f"\n Creating section summary...")

            # Prepare special prompt for section summary
            system_prompt = self._get_section_summary_prompt(subject_name, subject_type, previous_summary)

            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": all_content
                }
            ]

            # Call language model
            logger.info(f"Final prompt: {messages}")
            response = await self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.3,
                top_p=0.8
            )

            section_summary = response.choices[0].message.content

            logger.info(f" Section summary created successfully!")
            logger.info(f" Summary length: {len(section_summary)} characters")

            # Add title to section summary
            section_title = f"# סיכום פרק {section_id}\n\n"
            section_summary_with_title = section_title + section_summary

            # Save summary to blob in new structure: CourseID/section_summaries/SectionID.md
            summary_blob_path = f"{course_id}/section_summaries/{section_id}.md"

            logger.info(f"Saving section summary to blob: {summary_blob_path}")

            success = await blob_manager.upload_text_to_blob(
                text_content=section_summary_with_title,
                blob_name=summary_blob_path
            )

            if success:
                logger.info(f"Section summary saved to blob: {summary_blob_path}")
                return summary_blob_path
            else:
                logger.info(f" Failed to save section summary to blob")
                return None

        except Exception as e:
            logger.info(f"Error in section summarization: {str(e)}")
            return None

    async def summarize_course_from_blob(self, full_blob_path: str, subject_name: str = None,
                                         subject_type: str = None) -> str | None:
        """
        Summarize complete course from all section summary files in blob storage
        Args:
            full_blob_path: Path to section_summaries folder (e.g., "CS101/section_summaries")
            subject_name: Subject name for context
            subject_type: Subject type for context
        Returns:
            Summary path in blob or None if failed
        """

        try:
            # Parse path: "CS101/section_summaries" -> ["CS101", "section_summaries"]
            path_parts = full_blob_path.split('/')

            if len(path_parts) < 2:
                logger.info(f" Invalid path: {full_blob_path}. Should be in format: CourseID/section_summaries")
                return None

            course_id = path_parts[0]  # CS101
            # path_parts[1] should be section_summaries

            logger.info(f" CourseID: {course_id}")
            logger.info(f" section_summaries path: {full_blob_path}")

            # Create BlobManager with default container
            blob_manager = BlobManager()

            # Get list of all files in container
            all_files = await blob_manager.list_files()

            # Filter files in section_summaries folder
            sections_files = [f for f in all_files if f.startswith(full_blob_path + "/") and f.endswith(".md")]

            if not sections_files:
                logger.info(f" No section summary files found in {full_blob_path}")
                return None

            logger.info(f" Found {len(sections_files)} section summary files:")
            for file in sections_files:
                logger.info(f"  - {file}")

            # Download and read all files directly to memory
            all_content = ""
            successful_files = []

            for file_path in sections_files:
                logger.info(f"\n Downloading file to memory: {file_path}")

                try:
                    # Download file directly to memory
                    file_bytes = await blob_manager.download_to_memory(file_path)

                    if file_bytes:
                        # Convert to text
                        file_content = file_bytes.decode('utf-8')

                        if file_content.strip():
                            all_content += f"\n\n{'=' * 50}\n"
                            all_content += f"Section: {os.path.basename(file_path)}\n"
                            all_content += f"{'=' * 50}\n\n"
                            all_content += file_content
                            successful_files.append(file_path)
                            logger.info(f" File read successfully: {len(file_content)} characters")
                        else:
                            logger.info(f" Empty file: {file_path}")
                    else:
                        logger.info(f" Failed to download file: {file_path}")

                except Exception as e:
                    logger.info(f" Error processing file {file_path}: {e}")
                    continue

            if not successful_files:
                logger.info(f" Could not read any files from {full_blob_path}")
                return None

            logger.info(f"\n Total working with {len(successful_files)} files")
            logger.info(f" Total content length: {len(all_content)} characters")

            # Create summary
            logger.info(f"\n Creating complete course summary...")

            # Prepare special prompt for course summary
            system_prompt = self._get_course_summary_prompt(subject_name, subject_type)

            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": all_content
                }
            ]

            # Call language model
            response = await self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.3,
                top_p=0.8
            )

            course_summary = response.choices[0].message.content

            logger.info(f"Course summary created successfully!")
            logger.info(f" Summary length: {len(course_summary)} characters")

            # Add title to course summary
            if subject_name:
                course_title = f"# סיכום קורס {subject_name}\n\n"
            else:
                course_title = f"# סיכום קורס\n\n"
            course_summary_with_title = course_title + course_summary

            # Save summary to blob in new structure: CourseID/course_summary.md
            summary_blob_path = f"{course_id}/course_summary.md"

            logger.info(f" Saving course summary to blob: {summary_blob_path}")

            success = await blob_manager.upload_text_to_blob(
                text_content=course_summary_with_title,
                blob_name=summary_blob_path
            )

            if success:
                logger.info(f" Course summary saved to blob: {summary_blob_path}")
                return summary_blob_path
            else:
                logger.info(f" Failed to save course summary to blob")
                return None

        except Exception as e:
            logger.info(f" Error in course summarization: {str(e)}")
            return None


async def main():
    """Main function for testing all three types of summaries"""
    logger.info("Content Summarizer - Testing All Summary Types with Subject Parameters")
    logger.info("=" * 70)

    summarizer = ContentSummarizer()

    # Test parameters
    subject_name = "מתמטיקה בדידה"
    subject_type = "מתמטי"
    course_id = "Discrete_mathematics"
    section_id = "Section2"

    # # ========================================
    # # TEST 1: Individual File Summaries
    # # ========================================
    # logger.info("\n" + "=" * 70)
    # logger.info("TEST 1: Testing Individual File Summaries (summarize_md_file)")
    # logger.info("=" * 70)
    #
    # test_files = [
    #     f"{course_id}/{section_id}/Videos_md/2.md"
    #     # f"{course_id}/{section_id}/Docs_md/2002.md"
    # ]
    #
    # successful_files = 0
    # for i, blob_path in enumerate(test_files, 1):
    #     logger.info(f"\n--- File Test {i}/{len(test_files)} ---")
    #     logger.info(f"Testing file: {blob_path}")
    #     logger.info(f"Subject: {subject_name} ({subject_type})")
    #
    #     try:
    #         result = await summarizer.summarize_md_file(
    #             blob_path=blob_path,
    #             subject_name=subject_name,
    #             subject_type=subject_type
    #         )
    #
    #         if result:
    #             logger.info(f"File summary created successfully!")
    #             logger.info(f"Summary saved to: {result}")
    #             successful_files += 1
    #         else:
    #             logger.info(f"Failed to create file summary")
    #
    #     except Exception as e:
    #         logger.info(f" Error during file summarization: {str(e)}")
    #         traceback.print_exc()
    #
    #     if i < len(test_files):
    #         logger.info(" Waiting 3 seconds before next file...")
    #         await asyncio.sleep(3)
    #
    # logger.info(f"\nFile Summary Results: {successful_files}/{len(test_files)} successful")

    # ========================================
    # TEST 2: Section Summary
    # ========================================
    logger.info("\n" + "=" * 70)
    logger.info("TEST 2: Testing Section Summary (summarize_section_from_blob)")
    logger.info("=" * 70)

    section_path = f"{course_id}/{section_id}/file_summaries"
    logger.info(f"Testing section path: {section_path}")
    logger.info(f"Subject: {subject_name} ({subject_type})")


    try:
        logger.info("Waiting 5 seconds before section summary...")
        await asyncio.sleep(30)

        # נתיב לסיכום הקודם
        previous_section_path = f"{course_id}/section_summaries/Section1.md"
        logger.info(f"Using previous section summary path: {previous_section_path}")

        section_result = await summarizer.summarize_section_from_blob(
            full_blob_path=section_path,
            subject_name=subject_name,
            subject_type=subject_type,
            previous_summary_path=previous_section_path  # מעביר נתיב במקום טקסט
        )

        if section_result:
            logger.info(f"Section summary created successfully!")
            logger.info(f"Section summary saved to: {section_result}")
        else:
            logger.info(f"Failed to create section summary")
            logger.info(f"Make sure there are file summaries in: {section_path}")

    except Exception as e:
        logger.info(f"Error during section summarization: {str(e)}")
        traceback.print_exc()

    # # ========================================
    # # TEST 3: Course Summary
    # # ========================================
    # logger.info("\n" + "=" * 70)
    # logger.info("TEST 3: Testing Course Summary (summarize_course_from_blob)")
    # logger.info("=" * 70)
    #
    # course_path = f"{course_id}/section_summaries"
    # logger.info(f"Testing course path: {course_path}")
    # logger.info(f"Subject: {subject_name} ({subject_type})")
    #
    # try:
    #     logger.info(" Waiting 5 seconds before course summary...")
    #     await asyncio.sleep(30)
    #
    #     course_result = await summarizer.summarize_course_from_blob(
    #         full_blob_path=course_path,
    #         subject_name=subject_name,
    #         subject_type=subject_type
    #     )
    #
    #     if course_result:
    #         logger.info(f"Course summary created successfully!")
    #         logger.info(f"Course summary saved to: {course_result}")
    #     else:
    #         logger.info(f"Failed to create course summary")
    #         logger.info(f"Make sure there are section summaries in: {course_path}")
    #
    # except Exception as e:
    #     logger.info(f" Error during course summarization: {str(e)}")
    #     traceback.print_exc()

    # ========================================
    # FINAL SUMMARY
    # ========================================
    logger.info("\n" + "=" * 70)
    logger.info("ESTING COMPLETED - SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Course: {subject_name} ({subject_type})")
    logger.info(f"Course ID: {course_id}")
    logger.info(f"Section ID: {section_id}")
    logger.info("")
    logger.info("Tests Performed:")
    # logger.info(f"   1. Individual File Summaries: {successful_files}/{len(test_files)} successful")
    logger.info(f"   2. Section Summary: {'Correct' if 'section_result' in locals() and section_result else 'Error'}")
    # logger.info(f"   3. Course Summary: {'Correct' if 'course_result' in locals() and course_result else 'Error'}")
    logger.info("")
    logger.info("Note: Section and Course summaries depend on previous summaries existing in blob storage")
    logger.info("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
