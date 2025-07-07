"""
Advanced Document Search - Advanced search system for document chunks
Based on the index created with Index_docs.py
"""
import logging
from typing import List, Dict, Optional, Tuple
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery, VectorizableTextQuery
from openai import AzureOpenAI
import json

from config import (
    SEARCH_SERVICE_NAME, SEARCH_API_KEY,
    AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_EMBEDDING_MODEL
)

# Configure logging - only errors and warnings
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedDocumentSearch:
    """
    מערכת חיפוש מתקדמת למסמכים
    תומכת בחיפוש טקסטואלי, סמנטי ווקטורי
    """

    def __init__(self, index_name: str = "document-chunks"):
        self.index_name = index_name
        self.search_endpoint = f"https://{SEARCH_SERVICE_NAME}.search.windows.net"
        self.credential = AzureKeyCredential(SEARCH_API_KEY)

        # יצירת search client
        self.search_client = SearchClient(
            endpoint=self.search_endpoint,
            index_name=self.index_name,
            credential=self.credential
        )

        # יצירת OpenAI client לחיפוש וקטורי
        self.openai_client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )

        logger.info(f"AdvancedDocumentSearch initialized with index: {self.index_name}")

    def check_index_status(self) -> Dict:
        """בדיקת מצב האינדקס והצגת מידע בסיסי"""
        print("=" * 60)

        try:
            # חיפוש כללי לבדיקה
            results = self.search_client.search(
                search_text="*",
                select=["*"],
                top=3,
                include_total_count=True
            )

            total_count = results.get_count()
            docs = list(results)

            print(f"📊 סה\"כ צ'אנקים באינדקס: {total_count}")
            print(f"📄 מסמכים שהוחזרו לבדיקה: {len(docs)}")

            if docs:
                print(f"✅ האינדקס פעיל ומכיל נתונים")

                # הצגת 2 מסמכים לדוגמה
                print(f"\n📄 דוגמאות למסמכים באינדקס:")
                for i, doc in enumerate(docs[:2], 1):
                    print(f"\n📄 מסמך {i}:")
                    print(f"  🆔 ID: {doc.get('id', 'N/A')}")
                    print(f"  📄 Document ID: {doc.get('document_id', 'N/A')}")
                    print(f"  📝 Document Name: {doc.get('document_name', 'N/A')}")
                    print(f"  📑 Chunk Index: {doc.get('chunk_index', 'N/A')}")
                    print(f"  📋 Section Title: {doc.get('section_title', 'N/A')}")

                    # הצגת תוכן הטקסט
                    text = doc.get('text', '')
                    if text:
                        preview = text[:150] + "..." if len(text) > 150 else text
                        print(f"  📜 תוכן: {preview}")
                    print("-" * 30)

                # בדיקת וקטור
                vector = doc.get('vector')
                if vector:
                    print(f"  🔢 Vector: יש ({len(vector)} dimensions)")
                else:
                    print(f"  🔢 Vector: אין")

                # הצגת כל השדות הזמינים
                print(f"\n📋 שדות זמינים במסמך:")
                for key in sorted(doc.keys()):
                    if not key.startswith('@'):
                        value = doc[key]
                        if value is None:
                            print(f"    {key}: None")
                        elif isinstance(value, list):
                            print(f"    {key}: רשימה עם {len(value)} פריטים")
                        elif isinstance(value, str) and len(value) > 50:
                            print(f"    {key}: טקסט ({len(value)} תווים)")
                        else:
                            print(f"    {key}: {value}")

                return {
                    "status": "active",
                    "total_chunks": total_count,
                    "sample_doc": doc
                }
            else:
                print("⚠️ האינדקס קיים אבל ריק")
                return {"status": "empty", "total_chunks": 0}

        except Exception as e:
            print(f"❌ שגיאה בגישה לאינדקס: {e}")
            logger.error(f"Error checking index status: {e}")
            return {"status": "error", "error": str(e)}

    def simple_text_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """חיפוש טקסטואלי פשוט במידה ולא ניתן לחלץ embedding"""
        print("=" * 60)

        try:
            results = self.search_client.search(
                search_text=query,
                select=[
                    "id", "document_id", "document_name", "chunk_index", "section_title",
                    "text", "document_type", "created_date"
                ],
                top=top_k,
                include_total_count=True
            )

            docs = list(results)
            total_count = results.get_count()

            if not docs:
                print("❌ לא נמצאו תוצאות")
                return []

            print(f"✅ נמצאו {len(docs)} תוצאות מתוך {total_count} צ'אנקים:")

            for i, doc in enumerate(docs, 1):
                score = doc.get('@search.score', 0)
                print(f"\n📄 תוצאה {i} (ציון: {score:.3f}):")
                print(f"  📄 מסמך: {doc.get('document_name', 'לא זמין')} (ID: {doc.get('document_id', 'N/A')})")
                print(f"  📑 צ'אנק: {doc.get('chunk_index', 'N/A')}")
                section_title = doc.get('section_title', '')
                if section_title:
                    print(f"  📋 כותרת: {section_title}")

                text = doc.get('text', '')
                if text:
                    # הדגשת המילים שנמצאו (פשוט)
                    preview = text[:200] + "..." if len(text) > 200 else text
                    print(f"  📜 תוכן: {preview}")

                print("—" * 40)

            return docs

        except Exception as e:
            print(f"❌ שגיאה בחיפוש טקסטואלי: {e}")
            logger.error(f"Error in text search: {e}")
            return []

    def generate_query_embedding(self, query: str) -> List[float]:
        """יצירת embedding לשאלת החיפוש"""
        try:
            response = self.openai_client.embeddings.create(
                model=AZURE_OPENAI_EMBEDDING_MODEL,
                input=query
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            return []

    def hybrid_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """חיפוש היברידי - משלב טקסט ווקטור"""
        print("=" * 60)

        try:
            # יצירת embedding לשאלה
            query_vector = self.generate_query_embedding(query)
            if not query_vector:
                print("⚠️ לא ניתן ליצור embedding, מבצע חיפוש טקסטואלי בלבד")
                return self.simple_text_search(query, top_k)

            # חיפוש היברידי - using VectorizedQuery class and larger top for better blending
            results = self.search_client.search(
                search_text=query,
                vector_queries=[VectorizedQuery(
                    vector=query_vector,  # Use 'vector' parameter
                    k_nearest_neighbors=50,  # Larger for better blending
                    fields="vector"
                )],
                select=[
                    "id", "document_id", "document_name", "chunk_index", "section_title",
                    "text", "document_type", "created_date"
                ],
                top=50,  # Get more results for better blending, then slice client-side
                include_total_count=True
            )

            docs = list(results)
            total_count = results.get_count()

            if not docs:
                print("❌ לא נמצאו תוצאות היברידיות")
                return []

            # Slice to requested top_k for display and return
            docs = docs[:top_k]

            print(f"✅ נמצאו {len(docs)} תוצאות היברידיות מתוך {total_count} צ'אנקים:")

            for i, doc in enumerate(docs, 1):
                score = doc.get('@search.score', 0)
                print(f"\n📄 תוצאה {i} (ציון משולב: {score:.3f}):")
                print(f"  📄 מסמך: {doc.get('document_name', 'לא זמין')} (ID: {doc.get('document_id', 'N/A')})")
                print(f"  📑 צ'אנק: {doc.get('chunk_index', 'N/A')}")
                section_title = doc.get('section_title', '')
                if section_title:
                    print(f"  📋 כותרת: {section_title}")

                text = doc.get('text', '')
                if text:
                    preview = text[:200] + "..." if len(text) > 200 else text
                    print(f"  📜 תוכן: {preview}")

                print("—" * 40)

            return docs

        except Exception as e:
            print(f"❌ שגיאה בחיפוש היברידי: {e}")
            logger.error(f"Error in hybrid search: {e}")
            return []

    def semantic_search(self, query: str, top_k: int = 5, document_id: str = None) -> List[Dict]:
        """חיפוש סמנטי מתקדם"""
        print("=" * 60)

        try:
            # יצירת embedding לשאלה
            query_vector = self.generate_query_embedding(query)
            if not query_vector:
                print("⚠️ לא ניתן ליצור embedding, מבצע חיפוש טקסטואלי בלבד")
                return self.simple_text_search(query, top_k)

            # הכנת פרמטרים לחיפוש
            search_params = {
                "search_text": query,
                "query_type": "semantic",
                "semantic_configuration_name": "default",
                "query_language": "he-il",
                "highlight_fields": "text",
                "vector_queries": [VectorizedQuery(
                    vector=query_vector,
                    k_nearest_neighbors=top_k,
                    fields="vector"
                )],
                "select": [
                    "id", "document_id", "document_name", "chunk_index", "section_title",
                    "text", "document_type", "created_date"
                ],
                "top": top_k
            }

            # הוספת פילטר אם נדרש
            if document_id:
                escaped_document_id = document_id.replace("'", "''")
                search_params["filter"] = f"document_id eq '{escaped_document_id}'"

            # חיפוש סמנטי מתקדם
            results = self.search_client.search(**search_params)

            docs = list(results)

            if not docs:
                print("❌ לא נמצאו תוצאות סמנטיות")
                return []

            print(f"✅ נמצאו {len(docs)} תוצאות סמנטיות:")

            for i, doc in enumerate(docs, 1):
                score = doc.get('@search.score', 0)
                print(f"\n📄 תוצאה {i} (ציון סמנטי: {score:.3f}):")
                print(f"  📄 מסמך: {doc.get('document_name', 'לא זמין')} (ID: {doc.get('document_id', 'N/A')})")
                print(f"  📑 צ'אנק: {doc.get('chunk_index', 'N/A')}")
                section_title = doc.get('section_title', '')
                if section_title:
                    print(f"  📋 כותרת: {section_title}")

                text = doc.get('text', '')
                if text:
                    preview = text[:200] + "..." if len(text) > 200 else text
                    print(f"  📜 תוכן: {preview}")

                print("—" * 40)

            return docs

        except Exception as e:
            print(f"❌ שגיאה בחיפוש סמנטי מתקדם: {e}")
            logger.error(f"Error in semantic search: {e}")
            # Fallback to regular hybrid search
            return self.hybrid_search(query, top_k)

    def search_best_answers(self, query: str, k: int = 5, document_id: str = None) -> List[Dict]:
        """
        פונקציה פשוטה שמקבלת שאלה ומחזירה K התשובות הטובות ביותר
        משתמשת בחיפוש סמנטי כברירת מחדל, עם fallback להיברידי
        מחזירה רק את הנתונים ללא הדפסות

        Args:
            query: השאלה לחיפוש
            k: מספר התוצאות הטובות ביותר להחזיר
            document_id: אופציונלי - אם מוגדר, יחפש רק במסמך הספציפי הזה
        """
        try:
            # שימוש בחיפוש סמנטי שהוא הכי חכם
            results = self.semantic_search(query, k, document_id)
            return results
        except Exception:
            # fallback להיברידי אם הסמנטי נכשל
            results = self.hybrid_search(query, k)
            return results


def run_search_demo():
    """הרצת דמו מלא של מערכת החיפוש"""
    print("📄 מערכת חיפוש מתקדמת למסמכים")
    print("=" * 80)

    try:
        # יצירת מערכת החיפוש
        search_system = AdvancedDocumentSearch("document-chunks")

        # בדיקת מצב האינדקס
        print("\n🔧 בדיקת מצב האינדקס:")
        status = search_system.check_index_status()

        if status.get("status") != "active":
            print("❌ האינדקס לא פעיל או ריק. אנא ודא שהאינדקס נוצר ומכיל נתונים.")
            return

        # שאלות לדוגמה
        demo_queries = [
            "מה זה טרנזטיביות",
            "מתי יש שוויון בין מחלקות שקילות",
            "איך אפשר לשלול ביטוי"
        ]

        print(f"\n🎯 הרצת דמו עם {len(demo_queries)} שאלות:")

        for i, query in enumerate(demo_queries, 1):
            print(f"\n{'=' * 80}")
            print(f"🔢 שאלה {i} מתוך {len(demo_queries)}: '{query}'")
            print(f"{'=' * 80}")

            # השוואת שיטות חיפוש - קריאה ישירה לכל שיטה
            print(f"🎯 שאלה: '{query}'")
            print("=" * 80)

            # 1. חיפוש טקסטואלי פשוט
            print("\n1️⃣ חיפוש טקסטואלי פשוט:")
            print("-" * 40)
            search_system.simple_text_search(query, top_k=2)

            print("\n" + "=" * 80)

            # 2. חיפוש היברידי
            print("\n2️⃣ חיפוש היברידי:")
            print("-" * 40)
            search_system.hybrid_search(query, top_k=2)

            print("\n" + "=" * 80)

            # 3. חיפוש סמנטי מתקדם
            print("\n3️⃣ חיפוש סמנטי מתקדם:")
            print("-" * 40)
            search_system.semantic_search(query, top_k=2)

            print("\n" + "=" * 80)

            # 4. חיפוש סמנטי במסמך ספציפי
            print("\n4️⃣ חיפוש סמנטי במסמך ספציפי:")
            print("-" * 40)
            sample_document_id = "Ex5Sol"
            print(f"🎯 חיפוש במסמך: {sample_document_id}")
            search_system.semantic_search(query, top_k=2, document_id=sample_document_id)

            # הפסקה בין שאלות
            if i < len(demo_queries):
                print("\n" + "🔄 עובר לשאלה הבאה..." + "\n")

        print(f"\n🎉 דמו הושלם בהצלחה!")
        print("💡 ניתן להשתמש במערכת עם שאלות נוספות או לחפש לפי Document ID ספציפי")

    except Exception as e:
        print(f"❌ שגיאה בהרצת הדמו: {e}")
        logger.error(f"Error in demo: {e}")
        import traceback
        traceback.print_exc()


def main():
    """פונקציה ראשית"""
    try:
        run_search_demo()

        # יצירת מערכת החיפוש
        search_system = AdvancedDocumentSearch("document-chunks")

        # שאלה לדוגמה
        query = "מה ההגדרה של יחס שקילות"

        print(f"🔍 חיפוש עבור: '{query}'")
        print("=" * 60)

        # קריאה לפונקציה החדשה
        results = search_system.search_best_answers(query, k=5)
        # results = search_system.search_best_answers(query, k=5, document_id="Ex5Sol")

        if not results:
            print("❌ לא נמצאו תוצאות")
            return

        print(f"\n📋 {len(results)} התשובות הטובות ביותר:")
        print("=" * 60)

        for i, doc in enumerate(results, 1):
            score = doc.get('@search.score', 0)
            print(f"\n🏆 תשובה {i} (ציון: {score:.3f}):")
            print(f"  📄 מסמך: {doc.get('document_name', 'לא זמין')}")
            print(f"  📑 צ'אנק: {doc.get('chunk_index', 'N/A')}")
            section_title = doc.get('section_title', '')
            if section_title:
                print(f"  📋 כותרת: {section_title}")

            text = doc.get('text', '')
            if text:
                # הצגת הטקסט המלא של התשובה
                print(f"  💬 תוכן: {text}")

            print("-" * 40)

    except Exception as e:
        print(f"❌ שגיאה כללית: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
