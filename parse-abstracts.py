import xml.sax
import gzip
import io
import re
import time

# 5.5 million abstracts ~ 10 minutes

class AbstractHandler(xml.sax.ContentHandler):
    def __init__(self, output):
        self.inAbstract = False
        self.abstract = ""
        self.output = output
        self.counter = 0

    def startElement(self, name, attrs):
        if name == "abstract":
            self.counter += 1
            self.inAbstract = True

    def endElement(self, name):
        if self.inAbstract and name == "abstract":
            sanitized = re.sub(r'[^a-zA-Z0-9+-:;()?! ]+', '', self.abstract)
            sanitized = re.sub(r' +', ' ', sanitized)
            if not self.abstract.startswith("|") and len(sanitized) > 25:
                self.output.write(sanitized + "\n")
            self.abstract = ""
            self.inAbstract = False

            if self.counter % 10000 == 0:
                print('processed %d abstracts. process time: %.3f' % (self.counter, time.process_time()))

    def characters(self, content):
        if self.inAbstract:
            self.abstract += content

with gzip.open('enwiki-20180320-abstract.xml.gz', 'rb') as ingzip, gzip.open("abstract-per-line.txt.gz", "wt", encoding="utf-8") as outgzip:
    parser = xml.sax.make_parser()
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)
    parser.setContentHandler(AbstractHandler(outgzip))
    parser.parse(ingzip)

