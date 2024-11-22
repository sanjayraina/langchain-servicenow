import requests
import os
from typing import Iterator

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
import re

class ServiceNowLoader(BaseLoader):
    """An example document loader that reads a file line by line."""

    def __init__(self, table: str, query: str, fields: str) -> None:
        """Initialize the loader with a ServiceNow API details.

        Args:
            table: ServiceNow table, e.g. incident.
            query: encoded query to filter data
        """
        self.instance_url = os.environ['SERVICENOW_INSTANCE_URL']
        self.instance_username = os.environ['SERVICENOW_INSTANCE_USERNAME']
        self.instance_password = os.environ['SERVICENOW_INSTANCE_PASSWORD']
        self.table = table
        self.query = query
        self.fields = fields

    def lazy_load(self) -> Iterator[Document]:  # <-- Does not take any arguments
        """A lazy loader that reads table records line by line.

        When you're implementing lazy load methods, you should use a generator
        to yield documents one by one.
        """
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        url = self.instance_url + \
            "/api/now/table/" + self.table + "?sysparm_limit=10&sysparm_query=" + \
                self.query + "&sysparm_fields=" + self.fields
        response = requests.get(url, auth=(self.instance_username,
                                self.instance_password), headers=headers)
        data = response.json()
        
        line_number = 0
        fldarr = self.fields.split(',')
        pattern = re.compile('<.*?>')
        for rec in data['result']:
            doc = "\n"
            for fld in fldarr:
                doc +=  fld + ":" + rec[fld] + "\n"
            content = re.sub(pattern, '', doc)
            yield Document(page_content=content,
                metadata={"line_number": line_number, "source": self.instance_url},
            )
            line_number += 1