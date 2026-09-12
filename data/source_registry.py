from dataclasses import dataclass

@dataclass(frozen=True)
class Source:
    name: str
    url: str
    authority: str

SOURCES = {}

def register_source(source: Source) -> None:
    SOURCES[source.name] = source

def get_source(name: str):
    return SOURCES.get(name)
