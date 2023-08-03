#!/usr/bin/env python
# -*- coding:utf-8 -*-
import json
from typing import List
from universal_ie.task_format.task_format import TaskFormat
from universal_ie.utils import tokens_to_str
from universal_ie.ie_format import Entity, Label, Sentence, Span


"""
{
  "docid": "53f824d9dabfae938c7080d9_William E. Pelham_Department of Psychology", 
  "text": "Dr. Pelham is a 1970 graduate of Dartmouth College and earned his Ph.D. in Clinical Psychology from the State University of New York at Stony Brook in 1976. He was a faculty member at Washington State University, Florida State University, the University Pittsburgh (WPIC), and the State University of New York at Buffalo (SUNY Distinguished Professor and Director, Center for Children and Families) prior to moving to Florida International University in 2010, where he is Distinguished Professor of Psychology and Psychiatry, Director of the Center for Children and Families, and Chair of the Psychology Department. He remains an Emeritus SUNY Distinguished Professor at SUNY Buffalo and an Adjunct Professor of Psychiatry at WPIC. His summer treatment program for ADHD children has been recognized by Divisions 53 and 37 of the APA and by CHADD as a model program, is listed on the SAMHSA NREPP list, and is widely recognized as the state of the art in treatment for children and adolescents with Attention Deficit Hyperactivity Disorder (ADHD). Dr. Pelham has authored or co-authored more than 375 professional publications (h-index: 90; i10-index: 258) dealing with ADHD and its assessment and treatmentpsychosocial, pharmacological, and combinedand was named as one of the top 10 among the countrys 1,900 academic clinical psychologists in peer-reviewed publications (Stewart, Wu, & Roberts, 2007). Dr. Pelham is a fellow of the American Psychological Association and the American Psychological Society, and past President of the Society of Child Clinical and Adolescent Psychology (SCCAP) and the International Society for Research in Child and Adolescent Psychopathology (ISRCAP). He currently chairs the task force on Dissemination and Implementation of Evidence-based Practices for the SCCAP. He is a past recipient of the CHADD Hall of Fame award (2002), the SCCAP Career Achievement Award (2009), the Trailblazer Awards of the Families and Parenting SIG (2010) and ADHD SIG (2011) of ABCT, and most recently the 2012 lifetime achievement award from the Society for the Science of Clinical Psychology. He has held more than 75 research grants (15 current) from federal agencies (NIMH, NIAAA, NIDA, NINDS, NICHD, IES), foundations, and pharmaceutical companies. He has served as a consultant/advisor on ADHD and related topics to numerous federal agencies (NIMH, NIAAA, NIDA, IOM, OMAR, and the CDC) and organizations (AAP, AACAP, APA, CHADD). He founded and directs the biennial Niagara/Miami Conference on Evidence-based Treatments for Childhood and Adolescent Mental Health Problems.", 
  "tokens": ["Dr.", "Pelham", "is", "a", "1970", "graduate", "of", "Dartmouth", "College", "and", "earned", "his", "Ph.D.", "in", "Clinical", "Psychology", "from", "the", "State", "University", "of", "New", "York", "at", "Stony", "Brook", "in", "1976.", "He", "was", "a", "faculty", "member", "at", "Washington", "State", "University,", "Florida", "State", "University,", "the", "University", "Pittsburgh", "(WPIC),", "and", "the", "State", "University", "of", "New", "York", "at", "Buffalo", "(SUNY", "Distinguished", "Professor", "and", "Director,", "Center", "for", "Children", "and", "Families)", "prior", "to", "moving", "to", "Florida", "International", "University", "in", "2010,", "where", "he", "is", "Distinguished", "Professor", "of", "Psychology", "and", "Psychiatry,", "Director", "of", "the", "Center", "for", "Children", "and", "Families,", "and", "Chair", "of", "the", "Psychology", "Department.", "He", "remains", "an", "Emeritus", "SUNY", "Distinguished", "Professor", "at", "SUNY", "Buffalo", "and", "an", "Adjunct", "Professor", "of", "Psychiatry", "at", "WPIC.", "His", "summer", "treatment", "program", "for", "ADHD", "children", "has", "been", "recognized", "by", "Divisions", "53", "and", "37", "of", "the", "APA", "and", "by", "CHADD", "as", "a", "model", "program,", "is", "listed", "on", "the", "SAMHSA", "NREPP", "list,", "and", "is", "widely", "recognized", "as", "the", "state", "of", "the", "art", "in", "treatment", "for", "children", "and", "adolescents", "with", "Attention", "Deficit", "Hyperactivity", "Disorder", "(ADHD).", "Dr.", "Pelham", "has", "authored", "or", "co-authored", "more", "than", "375", "professional", "publications", "(h-index:", "90;", "i10-index:", "258)", "dealing", "with", "ADHD", "and", "its", "assessment", "and", "treatmentpsychosocial,", "pharmacological,", "and", "combinedand", "was", "named", "as", "one", "of", "the", "top", "10", "among", "the", "countrys", "1,900", "academic", "clinical", "psychologists", "in", "peer-reviewed", "publications", "(Stewart,", "Wu,", "&", "Roberts,", "2007).", "Dr.", "Pelham", "is", "a", "fellow", "of", "the", "American", "Psychological", "Association", "and", "the", "American", "Psychological", "Society,", "and", "past", "President", "of", "the", "Society", "of", "Child", "Clinical", "and", "Adolescent", "Psychology", "(SCCAP)", "and", "the", "International", "Society", "for", "Research", "in", "Child", "and", "Adolescent", "Psychopathology", "(ISRCAP).", "He", "currently", "chairs", "the", "task", "force", "on", "Dissemination", "and", "Implementation", "of", "Evidence-based", "Practices", "for", "the", "SCCAP.", "He", "is", "a", "past", "recipient", "of", "the", "CHADD", "Hall", "of", "Fame", "award", "(2002),", "the", "SCCAP", "Career", "Achievement", "Award", "(2009),", "the", "Trailblazer", "Awards", "of", "the", "Families", "and", "Parenting", "SIG", "(2010)", "and", "ADHD", "SIG", "(2011)", "of", "ABCT,", "and", "most", "recently", "the", "2012", "lifetime", "achievement", "award", "from", "the", "Society", "for", "the", "Science", "of", "Clinical", "Psychology.", "He", "has", "held", "more", "than", "75", "research", "grants", "(15", "current)", "from", "federal", "agencies", "(NIMH,", "NIAAA,", "NIDA,", "NINDS,", "NICHD,", "IES),", "foundations,", "and", "pharmaceutical", "companies.", "He", "has", "served", "as", "a", "consultant/advisor", "on", "ADHD", "and", "related", "topics", "to", "numerous", "federal", "agencies", "(NIMH,", "NIAAA,", "NIDA,", "IOM,", "OMAR,", "and", "the", "CDC)", "and", "organizations", "(AAP,", "AACAP,", "APA,", "CHADD).", "He", "founded", "and", "directs", "the", "biennial", "Niagara/Miami", "Conference", "on", "Evidence-based", "Treatments", "for", "Childhood", "and", "Adolescent", "Mental", "Health", "Problems."], 
  "ner": [[80, 83, "work_for"], [68, 69, "work_for"], [106, 111, "title"], [33, 34, "gender"], [13, 17, "highest_education"], [89, 95, "honorary_title"], [115, 122, "honorary_title"], [367, 369, "honorary_title"], [335, 341, "awards"], [345, 350, "awards"], [354, 363, "awards"], [0, 32, "education"], [33, 76, "work_record"], [112, 131, "work_record"], [428, 495, "work_record"], [268, 279, "take_office"], [281, 309, "take_office"]]
  }
"""


class En_bio(TaskFormat):
    def __init__(self, doc_json, language='en'):
        super().__init__(
            language=language
        )
        # self.doc_id = doc_json['docid']
        # self.sent_id = doc_json['text']
        self.tokens = doc_json["text"]
        self.entities = doc_json["entity"] # entity_mentions
        # self.relations = doc_json['relation_mentions']
        # self.events = doc_json['event_mentions']

    def generate_instance(self):
        entities = dict()

        for idex, span in enumerate(self.entities):
            span_start, span_end, entity, label = span[0], span[1], span[2], span[3] # start, end, entity, label
            tokens = self.tokens[span_start: span_end]
            assert tokens == entity, entity
            indexes = list(range(span_start, span_end))
            entities[(span_start, span_end, label)] = Entity(
                span=Span(
                    tokens=tokens,
                    indexes=indexes,
                    text=tokens_to_str(tokens, language=self.language),
                    
                ),
                label=Label(label)
            )

        return Sentence(
            tokens=self.tokens,
            entities=entities.values(),
            relations=list(),
            events=list()
        )

    @staticmethod
    def load_from_file(filename, language='en') -> List[Sentence]:
        sentence_list = list()
        with open(filename) as fin:
            for line in fin:
                instance = En_bio(
                    json.loads(line.strip()),
                    language=language
                ).generate_instance()
                sentence_list += [instance]
        return sentence_list
