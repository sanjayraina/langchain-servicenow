import os
import requests
from pydantic import BaseModel
import re

class ServiceNowAPIWrapper(BaseModel):
    """
    Wrapper around the ServiceNow API used to fetch table information.
    """
    doc_content_chars_max: int = 8000
    limit: int = 5
    url: str = os.environ['SERVICENOW_INSTANCE_URL']
    username: str = os.environ['SERVICENOW_INSTANCE_USERNAME']
    password: str = os.environ['SERVICENOW_INSTANCE_PASSWORD']
    fields: str = os.environ['SERVICENOW_FIELDS']
    filter: str = os.environ['SERVICENOW_FILTER']
        
    def get_incidents(self, query: str) -> str:
        """Run ServiceNow query and get page summaries."""
        if (query):
            sysparm_query = self.filter + "AND" + query
        else:
            sysparm_query = self.filter

        headers = {"Content-Type": "application/json",
                   "Accept": "application/json"}
        url = self.url + "/api/now/table/incident?sysparm_limit=" + \
            str(self.limit) + "&sysparm_query=" + sysparm_query + \
                "&sysparm_fields=" + self.fields
        response = requests.get(url, auth=(self.username, self.password), headers=headers)

        if response.status_code >= 200 and response.status_code < 300:
            textarr = []
            data = response.json()
            for rec in data['result']:
                fldarr = self.fields.split(',')
                for fld in fldarr:
                    pattern = re.compile('<.*?>')
                    text = rec[fld]
                    clear_text = re.sub(pattern, '', text)
                    rectext = f"{fld}: {clear_text}"
                    if ("undefined" not in rectext):
                        textarr.append(rectext)
            return "\n\n".join(textarr)[: self.doc_content_chars_max]            
        else:
            return f"Failed to call events API with status code {response.status_code}"
        
    def create_incident(self, query: str) -> str:
        """Create ServiceNow incident."""
        url = self.url + "/api/now/table/incident"
        headers = {"Content-Type": "application/json",
                   "Accept": "application/json"}
        response = requests.post(url, auth=(self.username, self.password), headers=headers, data=query)
        return response
        
        
