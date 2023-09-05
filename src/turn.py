# Copied from streamlit-webrtc/sample_utils/turn.py
import logging
import os

from twilio.rest import Client

logger = logging.getLogger(__name__)


def get_ice_servers():
    # Ref: https://www.twilio.com/docs/stun-turn/api
    try:
        account_sid = os.environ["TWILIO_ACCOUNT_SID"]
        auth_token = os.environ["TWILIO_AUTH_TOKEN"]
    except KeyError:
        logger.warning("Twilio credentials are not set. Fallback to a free STUN server from Google.")  # noqa: E501
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

    client = Client(account_sid, auth_token)

    token = client.tokens.create()

    return token.ice_servers


if __name__ == "__main__":
    print(get_ice_servers())
