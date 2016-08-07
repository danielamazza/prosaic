#!/usr/bin/env python
# This program is part of prosaic.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import TextIOBase
import logging
import re

import prosaic.cfg as cfg # TODO
import prosaic.nlp as nlp
from prosaic.models import Phrase, Source, Corpus, Session, Database, get_session # TODO

# ultimately, i want parsing to be as fast as possible so it can be reliably
# done online. it's painfully slow now and i want to try speeding this up
# before i start stressing about adding a job queue system or messing with
# websocket interfaces.

# it is probably going to be fastest to try and have as few runs over the data
# as possible. currently, there's a big regular expression pass, an nltk pass
# for sentences, then for each sentence, more passes for clause detection and
# cleanup.

# i'm currently wondering how feasible it is to do a finite state automata that
# cleans up bad characters as it goes and picks out clauses to put in the db.

# the process would look something like:

# stream book in chunks
# for each chunk, move character by character while in RAM
# if ok character or not clause or sentence marker, add to line buffer
# remove a bad character and skip
# if potential clause, check current line buffer length
# if line buffer seems full enough (ie a long enough line), submit to save pool
# if not, add to buffer
# if sentence marker, ensure buffer seems full enough and submit to save pool
# if not, wipe buffer

CHUNK_SIZE = 10000

BAD_CHARS = {'(': True,
             ')': True,
             '{': True,
             '}': True,
             '[': True,
             ']': True,
             '`': True,
             "'": True,
             '"': True,
             '\n': True,
             '“': True,
             '”': True,
             '«': True,
             '»': True,
             "'": True,
             '\\': True,
             '_': True,}
CLAUSE_MARKERS = {',':True, ';':True, ':':True}
SENTENCE_MARKERS = {'?':True, '.':True, '!':True}
# TODO random, magic number
LONG_ENOUGH = 20

def process_line(source_id: int, line_no: int, line: str) -> None:
    session = get_session(Database(**cfg.DEFAULT_DB))
    source = session.query(Source).filter(Source.id == source_id).one()

    stems = nlp.stem_sentence(line)
    rhyme_sound = nlp.rhyme_sound(line)
    syllables = nlp.count_syllables(line)
    alliteration = nlp.has_alliteration(line)

    phrase = Phrase(stems=stems, raw=line, alliteration=alliteration,
                    rhyme_sound=rhyme_sound,
                    syllables=syllables, line_no=line_no, source=source)

    session.add(phrase)
    session.commit()

def process_text_stream(source: Source, text: TextIOBase) -> None:
    session = Session.object_session(source)
    line_no = 0
    ultimate_text = ''
    futures = []
    source.content = ''
    session.add(source)
    session.commit() # so we can attach phrases to it. we'll commit again later.

    print('initializing pool')
    with ProcessPoolExecutor() as pool:
        print('reading chunk')
        chunk = text.read(CHUNK_SIZE)
        while len(chunk) > 0:
            line_buff = ""
            for c in chunk:
                if BAD_CHARS.get(c, False):
                    print('skipping bad character')
                    if not line_buff.endswith(" "):
                        line_buff += ' '
                    continue
                if CLAUSE_MARKERS.get(c, False):
                    if len(line_buff) > LONG_ENOUGH:
                        ultimate_text += line_buff
                        print('found clause, submitting')
                        print(line_buff)
                        futures.append(pool.submit(process_line, source.id, line_no, line_buff))
                        line_no += 1
                        line_buff = ""
                    else:
                        line_buff += c
                    continue
                if SENTENCE_MARKERS.get(c, False):
                    if len(line_buff) > LONG_ENOUGH:
                        ultimate_text += line_buff
                        print('found sentence, submitting')
                        print(line_buff)
                        futures.append(pool.submit(process_line, source.id, line_no, line_buff))
                        line_no += 1
                    line_buff = ""
                    continue
                line_buff += c
            print('reading chunk')
            chunk = text.read(CHUNK_SIZE)

        print('waiting on futures')
        for fut in as_completed(futures):
            if fut.exception() is not None:
                # TODO if error, cancel all futures and delete source.
                print('raising exception')
                raise fut.exception()

    print('process pool done, saving source')
    source.content = ultimate_text
    session.add(source)
    session.commit()
    session.close()

log = logging.getLogger('prosaic')

pairs = [('{', '}'), ('(', ')'), ('[', ']')]
bad_substrings = ['`', '“', '”', '«', '»', "''", '\\n', '\\',]
collapse_whitespace_re = re.compile("\s+")

def pre_process_text(raw_text: str) -> str:
    """Performs text-wide regex'ing we need before converting to sentences."""
    raw_text = re.sub(collapse_whitespace_re, ' ', raw_text)
    return raw_text

def pre_process_sentence(sentence: str) -> str:
    """Strip dangling pair characters. For now, strips some substrings that we
    don't want. r and lstrip. Returns modified sentence"""
    if sentence.count('"') == 1:
        sentence = sentence.replace('"', '')

    # TODO bootleg
    for l,r in pairs:
        if sentence.count(l) == 1 and sentence.count(r) == 0:
            sentence = sentence.replace(l, '')
        if sentence.count(r) == 1 and sentence.count(l) == 0:
            sentence = sentence.replace(r, '')

    # TODO collapse this into a regex and do it in pre_process_text
    for substring in bad_substrings:
       sentence = sentence.replace(substring, '')

    return sentence.rstrip().lstrip()

def process_text(source: Source, raw_text: str) -> None:
    """Given raw text and a source filename, adds a new source with the raw
    text as its content and then processes all of the phrases in the text."""

    log.debug('connecting to db...')
    session = Session.object_session(source)

    log.debug('pre-processing text...')
    text = pre_process_text(raw_text)

    log.debug('adding source to corpus...')
    source.content = text
    session.add(source)

    log.debug('extracting sentences')
    sentences = nlp.sentences(text)

    # log.debug("expanding clauses...")
    # sentences = nlp.expand_multiclauses(sentences)

    log.debug("pre-processing, parsing and saving sentences...")
    for x in range(0, len(sentences)):
        sentence = pre_process_sentence(sentences[x])

        stems = nlp.stem_sentence(sentence)
        rhyme_sound = nlp.rhyme_sound(sentence)
        syllables = nlp.count_syllables(sentence)
        alliteration = nlp.has_alliteration(sentence)

        phrase = Phrase(stems=stems, raw=sentence, alliteration=alliteration,
                        rhyme_sound=rhyme_sound,
                        syllables=syllables, line_no=x, source=source)

        session.add(phrase)

    log.debug("done processing text; changes not yet committed")
