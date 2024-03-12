from .common import *
from .datasets import *
from .models import *
from .tasks import *
from .runners import *

# override default path
import os
from lavis.common.registry import registry

repo_path = os.path.abspath("./")
registry.mapping['paths']['library_root'] = repo_path
registry.mapping['paths']['repo_root'] = repo_path
registry.mapping['paths']['cache_root'] = repo_path

