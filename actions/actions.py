# import httpx 
from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from sql_helper import qp

class ActionSearchTable(Action):

    def name(self) -> Text:
        return "action_search_table"

    async def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        # async with httpx.AsyncClient() as client:
        #     json_data = {"data": [{"mime_type": "text/plain", "text": tracker.latest_message['text']}]}
        #     resp = ""#await client.post("http://localhost:12345/search", json=json_data)
        try:
            resp = qp.run(
                        query=tracker.latest_message['text']
                    ).message.content

            # dispatcher.utter_message(text="I found these results:")
            dispatcher.utter_message(text=resp)
        except:
            dispatcher.utter_message(text="It appears that there are results for your query. Kindly finetune your query so that it matches, If you have any other questions or need assistance with something else, feel free to ask!")
        # matches = resp.json()['data']['docs'][0]['matches']
        # for m in matches[:5]:
        #     dispatcher.utter_message(text=f"- {m['text']}")

        return []
