#!/usr/bin/env python
# -*- coding:utf-8 -*-
import json
from typing import List
from universal_ie.task_format.task_format import TaskFormat
from universal_ie.utils import tokens_to_str
from universal_ie.ie_format import Entity, Event, Label, Sentence, Span


"""

[
  ["March_2012 Nazi war criminal John Demjanjuk, convicted of accessory to the murder of 27,900 Jews during The Holocaust, dies at the age of 91. (BBC)", "John Demjanjuk, who was found guilty for his role as a guard at a Nazi death camp in World War II, has died aged 91, German police say.\nHe had been sentenced in May 2011 by a German court to five years in prison, but was released pending an appeal.\nHe died at a home for the elderly.\nThe court said Demjanjuk, 91, was a guard at Sobibor camp in Nazi-occupied Poland in 1943. He denied this, saying he was a prisoner of war and a victim too.\nAn estimated 250,000 people died in the gas chambers at Sobibor. Demjanjuk was convicted of being an accessory to the murder of the 28,060 people who were killed there while he was a guard.\nDemjanjuk's family said during his trial that he was very ill.\nHe was also convicted on similar charges by a court in Israel in 1986, but the verdict was overturned when doubts emerged about his identity.\nBorn in Ukraine in 1920, Demjanjuk grew up under Soviet rule.\nHe was a soldier in the Red Army in 1942 when he was captured by the Germans.\nProsecutors had argued he was recruited by the Germans to be an SS camp guard and that by working at a death camp he was a participant in the killings. No evidence was produced that he committed a specific crime.\nIt was the first time such a legal argument had been made in a German court.\nCentral to the prosecution's case was an SS identity card indicating Demjanjuk had been posted to Sobibor. The defence cast doubts on the authenticity of the card but court experts said it appeared genuine.", "Famous Person - Death", 
  [{"start": 113, "end": 115, "type": "Age", "text": "91"}, {"start": 55, "end": 60, "type": "Profession", "text": "guard"}, {"start": 0, "end": 14, "type": "Deceased", "text": "John Demjanjuk"}, {"start": 260, "end": 282, "type": "Location/Hospital", "text": "a home for the elderly"}]
  ],
  [
    ...
  ]]
"""


class DOCEvent(TaskFormat):
    def __init__(self, doc_json, language='en'):
        super().__init__(
            language=language
        )

        self.title = doc_json[0]
        self.text = doc_json[1]
        self.event_type = doc_json[2]
        self.events = doc_json[3]

    def generate_instance(self):
        events = dict()
        entities = dict()

        for span_index, span in enumerate(self.entities):
            tokens = self.tokens[span['start']: span['end']]
            indexes = list(range(span['start'], span['end']))
            entities[span['id']] = Entity(
                span=Span(
                    tokens=tokens,
                    indexes=indexes,
                    text=tokens_to_str(tokens, language=self.language),
                    text_id=self.sent_id
                ),
                label=Label(span['entity_type']),
                text_id=self.sent_id,
                record_id=span['id']
            )

        for event_index, event in enumerate(self.events):
            start = event['trigger']['start']
            end = event['trigger']['end']
            tokens = self.tokens[start:end]
            indexes = list(range(start, end))
            events[event['id']] = Event(
                span=Span(
                    tokens=tokens,
                    indexes=indexes,
                    text=tokens_to_str(tokens, language=self.language),
                    text_id=self.sent_id
                ),
                label=Label(event['event_type']),
                args=[(Label(x['role']), entities[x['entity_id']])
                      for x in event['arguments']],
                text_id=self.sent_id,
                record_id=event['id']
            )

        return Sentence(
            tokens=self.tokens,
            entities=list(),
            relations=list(),
            events=events.values(),
            text_id=self.sent_id
        )

    @staticmethod
    def load_from_file(filename, language='en') -> List[Sentence]:
        sentence_list = list()
        with open(filename) as fin:
            for line in fin:
                instance = DOCEvent(
                    json.loads(line.strip()),
                    language=language
                ).generate_instance()
                sentence_list += [instance]
        return sentence_list
