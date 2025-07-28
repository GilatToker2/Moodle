from fastapi import FastAPI, HTTPException
import os
from dotenv import load_dotenv
from VideoIndexerClient.Consts import Consts
from VideoIndexerClient.VideoIndexerClient import VideoIndexerClient

app = FastAPI(title="Video Indexer API")

# Load configuration
load_dotenv()
consts = Consts(
    ApiVersion='2024-01-01',
    ApiEndpoint='https://api.videoindexer.ai',
    AzureResourceManager='https://management.azure.com',
    AccountName=os.getenv('VIDEO_INDEXER_VI_ACC'),
    ResourceGroup=os.getenv('VIDEO_INDEXER_RG'),
    SubscriptionId=os.getenv('VIDEO_INDEXER_SUB_ID')
)

# Initialize Video Indexer Client
client = VideoIndexerClient()
arm_access_token_, vi_access_token_, full_response = client.authenticate_async(consts)

@app.get("/account")
async def get_account():
    """Get Video Indexer account details"""
    try:
        client.get_account_async()
        return {"account_id": client.account["properties"]["accountId"],
                "location": client.account["location"],
                "arm_access_token_": arm_access_token_,
                "vi_access_token_": vi_access_token_,
                "full_response": full_response}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)