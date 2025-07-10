#!/usr/bin/env python3
"""
Simple test to verify LiveKit connection
"""

import asyncio
import time
from livekit import rtc

def generate_token(identity: str, room_name: str) -> str:
    """Generate a JWT token for LiveKit authentication"""
    try:
        import jwt
        
        # Token payload
        payload = {
            "video": {"room": room_name, "roomJoin": True},
            "iss": "devkey",
            "sub": identity,
            "exp": int(time.time()) + 3600,  # 1 hour
            "iat": int(time.time())
        }
        
        # Generate token
        token = jwt.encode(payload, "secret", algorithm="HS256")
        return token
        
    except ImportError:
        print("⚠️  PyJWT not installed. Install with: pip install PyJWT")
        # Fallback for testing - this won't work but will show the error
        return "dev-token"

async def test_connection():
    print("🔍 Testing LiveKit connection...")
    
    try:
        # Create room
        room = rtc.Room()
        
        # Generate proper token
        token = generate_token("test-user", "test-room")
        
        # Try to connect
        await room.connect(
            "ws://localhost:7880",
            token=token
        )
        
        print("✅ Successfully connected to LiveKit!")
        print(f"📊 Room SID: {room.sid}")
        print(f" Local participant: {room.local_participant.identity}")
        
        # Disconnect
        await room.disconnect()
        print("👋 Disconnected successfully")
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("💡 Make sure LiveKit server is running: docker-compose up -d")

if __name__ == "__main__":
    asyncio.run(test_connection()) 