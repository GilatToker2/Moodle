"""
Advanced Video Search - מערכת חיפוש מתקדמת לתמלולי וידאו
מבוססת על האינדקס שנוצר עם Index_transcription.py
משתמשת בקבצי הדיבג הקיימים כהשראה
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AdvancedVideoSearch:
    """
    מערכת חיפוש מתקדמת לתמלולי וידאו
    תומכת בחיפוש טקסטואלי, סמנטי ווקטורי
    """

    def __init__(self, index_name: str = "video-chunks"):
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

        logger.info(f"AdvancedVideoSearch initialized with index: {self.index_name}")

    def check_index_status(self) -> Dict:
        """בדיקת מצב האינדקס והצגת מידע בסיסי"""
        logger.info("🔍 בדיקת מצב האינדקס")
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

                # הצגת מסמך לדוגמה
                doc = docs[0]
                print(f"\n📄 דוגמה למסמך באינדקס:")
                print(f"  🆔 ID: {doc.get('id', 'N/A')}")
                print(f"  🎬 Video ID: {doc.get('video_id', 'N/A')}")
                print(f"  📝 Video Name: {doc.get('video_name', 'N/A')}")
                print(f"  ⏰ Start Time: {doc.get('start_time', 'N/A')}")
                print(f"  ⏱️ Start Seconds: {doc.get('start_seconds', 'N/A')}")

                # הצגת תוכן הטקסט
                text = doc.get('text', '')
                if text:
                    preview = text[:150] + "..." if len(text) > 150 else text
                    print(f"  📜 תוכן: {preview}")

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
        """חיפוש טקסטואלי פשוט"""
        logger.info(f"🔍 חיפוש טקסטואלי: '{query}'")
        print("=" * 60)

        try:
            results = self.search_client.search(
                search_text=query,
                select=[
                    "id", "video_id", "video_name", "start_time", "start_seconds",
                    "text", "created_date"
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
                print(f"  🎬 וידאו: {doc.get('video_name', 'לא זמין')} (ID: {doc.get('video_id', 'N/A')})")
                print(f"  ⏰ זמן: {doc.get('start_time', 'N/A')} ({doc.get('start_seconds', 0):.1f}s)")

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

    def vector_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """חיפוש וקטורי סמנטי"""
        logger.info(f"🔍 חיפוש וקטורי: '{query}'")
        print("=" * 60)

        try:
            # יצירת embedding לשאלה
            query_vector = self.generate_query_embedding(query)
            if not query_vector:
                print("❌ לא ניתן ליצור embedding לשאלה")
                return []

            # חיפוש וקטורי - using VectorizedQuery class
            results = self.search_client.search(
                search_text=None,
                vector_queries=[VectorizedQuery(
                    vector=query_vector,  # Use 'vector' parameter
                    k_nearest_neighbors=top_k,
                    fields="vector"
                )],
                select=[
                    "id", "video_id", "video_name", "start_time", "start_seconds",
                    "text", "created_date"
                ],
                top=top_k
            )

            docs = list(results)

            if not docs:
                print("❌ לא נמצאו תוצאות וקטוריות")
                return []

            print(f"✅ נמצאו {len(docs)} תוצאות וקטוריות:")

            for i, doc in enumerate(docs, 1):
                score = doc.get('@search.score', 0)
                print(f"\n📄 תוצאה {i} (דמיון: {score:.3f}):")
                print(f"  🎬 וידאו: {doc.get('video_name', 'לא זמין')} (ID: {doc.get('video_id', 'N/A')})")
                print(f"  ⏰ זמן: {doc.get('start_time', 'N/A')} ({doc.get('start_seconds', 0):.1f}s)")

                text = doc.get('text', '')
                if text:
                    preview = text[:200] + "..." if len(text) > 200 else text
                    print(f"  📜 תוכן: {preview}")

                print("—" * 40)

            return docs

        except Exception as e:
            print(f"❌ שגיאה בחיפוש וקטורי: {e}")
            logger.error(f"Error in vector search: {e}")
            return []

    def hybrid_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """חיפוש היברידי - משלב טקסט ווקטור"""
        logger.info(f"🔍 חיפוש היברידי: '{query}'")
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
                    "id", "video_id", "video_name", "start_time", "start_seconds",
                    "text", "created_date"
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
                print(f"  🎬 וידאו: {doc.get('video_name', 'לא זמין')} (ID: {doc.get('video_id', 'N/A')})")
                print(f"  ⏰ זמן: {doc.get('start_time', 'N/A')} ({doc.get('start_seconds', 0):.1f}s)")

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

    def semantic_search_with_captions(self, query: str, top_k: int = 5) -> List[Dict]:
        """חיפוש סמנטי מתקדם עם captions אוטומטיים"""
        logger.info(f"🔍 חיפוש סמנטי מתקדם: '{query}'")
        print("=" * 60)

        try:
            # יצירת embedding לשאלה
            query_vector = self.generate_query_embedding(query)
            if not query_vector:
                print("⚠️ לא ניתן ליצור embedding, מבצע חיפוש טקסטואלי בלבד")
                return self.simple_text_search(query, top_k)

            # חיפוש סמנטי מתקדם עם captions - fallback to basic semantic search
            try:
                results = self.search_client.search(
                    search_text=query,
                    query_type="semantic",
                    semantic_configuration_name="default",
                    query_language="he-il",  # Required for captions
                    captions="extractive",  # Auto-generated summaries
                    highlight_fields="text",
                    vector_queries=[VectorizedQuery(
                        vector=query_vector,  # Use 'vector' parameter
                        k_nearest_neighbors=top_k,
                        fields="vector"
                    )],
                    select=[
                        "id", "video_id", "video_name", "start_time", "start_seconds",
                        "text", "created_date", "@search.captions"
                    ],
                    top=top_k
                )
            except Exception as captions_error:
                logger.warning(f"Captions not supported, falling back to basic semantic search: {captions_error}")
                # Fallback to semantic search without captions
                results = self.search_client.search(
                    search_text=query,
                    query_type="semantic",
                    semantic_configuration_name="default",
                    highlight_fields="text",
                    vector_queries=[VectorizedQuery(
                        vector=query_vector,  # Use 'vector' parameter
                        k_nearest_neighbors=top_k,
                        fields="vector"
                    )],
                    select=[
                        "id", "video_id", "video_name", "start_time", "start_seconds",
                        "text", "created_date"
                    ],
                    top=top_k
                )

            docs = list(results)

            if not docs:
                print("❌ לא נמצאו תוצאות סמנטיות")
                return []

            print(f"✅ נמצאו {len(docs)} תוצאות סמנטיות:")

            for i, doc in enumerate(docs, 1):
                score = doc.get('@search.score', 0)
                print(f"\n📄 תוצאה {i} (ציון סמנטי: {score:.3f}):")
                print(f"  🎬 וידאו: {doc.get('video_name', 'לא זמין')} (ID: {doc.get('video_id', 'N/A')})")
                print(f"  ⏰ זמן: {doc.get('start_time', 'N/A')} ({doc.get('start_seconds', 0):.1f}s)")

                # הצגת captions אוטומטיים אם קיימים
                captions = doc.get("@search.captions")
                if captions and len(captions) > 0:
                    print(f"  📝 סיכום אוטומטי: {captions[0].get('text', '')}")

                text = doc.get('text', '')
                if text:
                    preview = text[:200] + "..." if len(text) > 200 else text
                    print(f"  📜 תוכן מלא: {preview}")

                print("—" * 40)

            return docs

        except Exception as e:
            print(f"❌ שגיאה בחיפוש סמנטי מתקדם: {e}")
            logger.error(f"Error in semantic search with captions: {e}")
            # Fallback to regular hybrid search
            return self.hybrid_search(query, top_k)

    def search_by_video_id(self, video_id: str, top_k: int = 10) -> List[Dict]:
        """חיפוש לפי video ID ספציפי"""
        logger.info(f"🔍 חיפוש לפי Video ID: '{video_id}'")
        print("=" * 60)

        try:
            # Escape single quotes in video_id for OData filter
            escaped_video_id = video_id.replace("'", "''")

            results = self.search_client.search(
                search_text="*",
                filter=f"video_id eq '{escaped_video_id}'",
                select=[
                    "id", "video_id", "video_name", "start_time", "start_seconds",
                    "text", "created_date"
                ],
                order_by=["start_seconds asc"],
                top=top_k,
                include_total_count=True
            )

            docs = list(results)
            total_count = results.get_count()

            if not docs:
                print(f"❌ לא נמצאו צ'אנקים עבור Video ID: {video_id}")
                return []

            print(f"✅ נמצאו {len(docs)} צ'אנקים עבור Video ID {video_id} (מתוך {total_count} סה\"כ):")

            for i, doc in enumerate(docs, 1):
                print(f"\n📄 צ'אנק {i}:")
                print(f"  🆔 ID: {doc.get('id', 'N/A')}")
                print(f"  ⏰ זמן: {doc.get('start_time', 'N/A')} ({doc.get('start_seconds', 0):.1f}s)")

                text = doc.get('text', '')
                if text:
                    preview = text[:150] + "..." if len(text) > 150 else text
                    print(f"  📜 תוכן: {preview}")

                print("—" * 30)

            return docs

        except Exception as e:
            print(f"❌ שגיאה בחיפוש לפי Video ID: {e}")
            logger.error(f"Error in video ID search: {e}")
            return []

    def compare_search_methods(self, query: str, top_k: int = 3) -> Dict:
        """השוואה בין שיטות חיפוש שונות"""
        logger.info(f"🔬 השוואת שיטות חיפוש עבור: '{query}'")
        print("=" * 80)

        results = {}

        # 1. חיפוש טקסטואלי
        print("\n1️⃣ חיפוש טקסטואלי:")
        print("-" * 40)
        results['text'] = self.simple_text_search(query, top_k)

        # 2. חיפוש וקטורי
        print("\n2️⃣ חיפוש וקטורי:")
        print("-" * 40)
        results['vector'] = self.vector_search(query, top_k)

        # 3. חיפוש היברידי
        print("\n3️⃣ חיפוש היברידי:")
        print("-" * 40)
        results['hybrid'] = self.hybrid_search(query, top_k)

        # 4. חיפוש סמנטי מתקדם
        print("\n4️⃣ חיפוש סמנטי מתקדם:")
        print("-" * 40)
        results['semantic'] = self.semantic_search_with_captions(query, top_k)

        # סיכום השוואה
        print(f"\n📊 סיכום השוואה עבור: '{query}'")
        print("=" * 60)
        for method, docs in results.items():
            method_name = {
                'text': 'טקסטואלי',
                'vector': 'וקטורי',
                'hybrid': 'היברידי',
                'semantic': 'סמנטי מתקדם'
            }.get(method, method.capitalize())

            print(f"{method_name}: {len(docs)} תוצאות")
            if docs:
                avg_score = sum(doc.get('@search.score', 0) for doc in docs) / len(docs)
                print(f"  ציון ממוצע: {avg_score:.3f}")

                # הצגת התוצאה הטובה ביותר
                best_doc = docs[0]
                print(f"  תוצאה מובילה: {best_doc.get('video_name', 'N/A')} - {best_doc.get('start_time', 'N/A')}")

        return results

    def unified_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """חיפוש מאוחד שמשלב את כל הטכנולוגיות - טקסט, וקטור וסמנטי"""
        logger.info(f"🔍 חיפוש מאוחד מתקדם: '{query}'")
        print("=" * 60)

        try:
            # חיפוש מאוחד עם כל הטכנולוגיות
            results = self.search_client.search(
                search_text=query,
                vector_queries=[VectorizableTextQuery(
                    text=query,
                    k_nearest_neighbors=50,
                    fields="vector"
                )],
                top=top_k,
                select=[
                    "id", "video_id", "video_name", "start_time", "start_seconds",
                    "text", "created_date"
                ],
                query_type="semantic",
                semantic_configuration_name="default",
                query_language="he-il"
            )

            docs = list(results)

            if not docs:
                print("❌ לא נמצאו תוצאות")
                return []

            print(f"✅ נמצאו {len(docs)} תוצאות מאוחדות:")

            for i, doc in enumerate(docs, 1):
                score = doc.get('@search.score', 0)
                print(f"\n📄 תוצאה {i} (ציון מאוחד: {score:.3f}):")
                print(f"  🎬 וידאו: {doc.get('video_name', 'לא זמין')} (ID: {doc.get('video_id', 'N/A')})")
                print(f"  ⏰ זמן: {doc.get('start_time', 'N/A')} ({doc.get('start_seconds', 0):.1f}s)")

                text = doc.get('text', '')
                if text:
                    preview = text[:200] + "..." if len(text) > 200 else text
                    print(f"  📜 תוכן: {preview}")

                print("—" * 40)

            return docs

        except Exception as e:
            print(f"❌ שגיאה בחיפוש מאוחד: {e}")
            logger.error(f"Error in unified search: {e}")
            # Fallback to simple text search
            print("🔄 חוזר לחיפוש טקסטואלי פשוט...")
            return self.simple_text_search(query, top_k)

    def get_video_summary(self, video_id: str) -> Dict:
        """קבלת סיכום של וידאו ספציפי"""
        logger.info(f"📊 יצירת סיכום עבור Video ID: {video_id}")

        try:
            # קבלת כל הצ'אנקים של הוידאו
            results = self.search_client.search(
                search_text="*",
                filter=f"video_id eq '{video_id}'",
                select=["video_name", "start_seconds", "text", "created_date"],
                order_by=["start_seconds asc"],
                top=1000  # מספר גבוה לקבלת כל הצ'אנקים
            )

            docs = list(results)

            if not docs:
                return {"error": f"לא נמצא וידאו עם ID: {video_id}"}

            # חישוב סטטיסטיקות
            total_chunks = len(docs)
            video_name = docs[0].get('video_name', 'לא זמין')
            total_text_length = sum(len(doc.get('text', '')) for doc in docs)

            # זמן כולל (משוער)
            if docs:
                max_time = max(doc.get('start_seconds', 0) for doc in docs)
                duration_minutes = max_time / 60
            else:
                duration_minutes = 0

            summary = {
                "video_id": video_id,
                "video_name": video_name,
                "total_chunks": total_chunks,
                "estimated_duration_minutes": round(duration_minutes, 2),
                "total_text_characters": total_text_length,
                "average_chunk_length": round(total_text_length / total_chunks) if total_chunks > 0 else 0,
                "created_date": docs[0].get('created_date') if docs else None
            }

            print(f"📊 סיכום וידאו {video_id}:")
            print(f"  📝 שם: {summary['video_name']}")
            print(f"  📄 צ'אנקים: {summary['total_chunks']}")
            print(f"  ⏱️ משך משוער: {summary['estimated_duration_minutes']} דקות")
            print(f"  📏 אורך טקסט כולל: {summary['total_text_characters']} תווים")
            print(f"  📊 אורך צ'אנק ממוצע: {summary['average_chunk_length']} תווים")

            return summary

        except Exception as e:
            logger.error(f"Error creating video summary: {e}")
            return {"error": str(e)}


def run_search_demo():
    """הרצת דמו מלא של מערכת החיפוש"""
    print("🎬 מערכת חיפוש מתקדמת לתמלולי וידאו")
    print("=" * 80)

    try:
        # יצירת מערכת החיפוש
        search_system = AdvancedVideoSearch("video-chunks")

        # בדיקת מצב האינדקס
        print("\n🔧 בדיקת מצב האינדקס:")
        status = search_system.check_index_status()

        if status.get("status") != "active":
            print("❌ האינדקס לא פעיל או ריק. אנא ודא שהאינדקס נוצר ומכיל נתונים.")
            return

        # שאלות לדוגמה
        demo_queries = [
            "מה זה פסוק לוגי",
            "מבוא ללוגיקה",
            "טענה"
        ]

        print(f"\n🎯 הרצת דמו עם {len(demo_queries)} שאלות:")

        for i, query in enumerate(demo_queries, 1):
            print(f"\n{'='*80}")
            print(f"🔢 שאלה {i} מתוך {len(demo_queries)}: '{query}'")
            print(f"{'='*80}")

            # השוואת שיטות חיפוש
            results = search_system.compare_search_methods(query, top_k=2)

            # הפסקה בין שאלות
            if i < len(demo_queries):
                print("\n" + "🔄 עובר לשאלה הבאה..." + "\n")

        # דוגמה לחיפוש לפי Video ID
        if status.get("sample_doc"):
            sample_video_id = status["sample_doc"].get("video_id")
            if sample_video_id:
                print(f"\n{'='*80}")
                print(f"📹 דוגמה: חיפוש לפי Video ID")
                print(f"{'='*80}")

                search_system.search_by_video_id(sample_video_id, top_k=5)

                # סיכום הוידאו
                print(f"\n📊 סיכום הוידאו:")
                summary = search_system.get_video_summary(sample_video_id)

        print(f"\n🎉 דמו הושלם בהצלחה!")
        print("💡 ניתן להשתמש במערכת עם שאלות נוספות או לחפש לפי Video ID ספציפי")

    except Exception as e:
        print(f"❌ שגיאה בהרצת הדמו: {e}")
        logger.error(f"Error in demo: {e}")
        import traceback
        traceback.print_exc()


def main():
    """פונקציה ראשית"""
    try:
        run_search_demo()
    except Exception as e:
        print(f"❌ שגיאה כללית: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
