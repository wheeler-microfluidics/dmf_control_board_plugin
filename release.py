import re
import sys
import subprocess
import tarfile
import yaml

# http://www.python.org/dev/peps/pep-0386/
_PEP386_SHORT_VERSION_RE = r'\d+(?:\.\d+)+(?:(?:[abc]|rc)\d+(?:\.\d+)*)?'
_PEP386_VERSION_RE = r'^%s(?:\.post\d+)?(?:\.dev\d+)?$' % (
    _PEP386_SHORT_VERSION_RE)
_GIT_DESCRIPTION_RE = r'^v(?P<ver>%s)-(?P<commits>\d+)-g(?P<sha>[\da-f]+)$' % (
    _PEP386_SHORT_VERSION_RE)

package_name = 'dmf_control_board_plugin'
plugin_name = 'wheelerlab.dmf_control_board_plugin'

# create a version sting based on the git revision/branch
def readGitVersion():
    try:
        proc = subprocess.Popen(('git', 'describe', '--long',
                                 '--match', 'v[0-9]*.*'),
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        data, _ = proc.communicate()
        if proc.returncode:
            return None
        ver = data.splitlines()[0].strip()
        proc = subprocess.Popen(('git', 'rev-parse', '--abbrev-ref', 'HEAD'),
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        branch, _ = proc.communicate()
        if proc.returncode:
            return None
    except:
        return None

    if not ver:
        return None
    m = re.search(_GIT_DESCRIPTION_RE, ver)
    if not m:
        sys.stderr.write('version: git description (%s) is invalid, '
                         'ignoring\n' % ver)
        return None

    commits = int(m.group('commits'))

    if not commits:
        version = m.group('ver')
    else:
        version = '%s.post%d' % (
            m.group('ver'), commits)

    if branch.strip() != 'master' and not branch.startswith('release'):
        version += '.dev%d' % int(m.group('sha'), 16)

    # Ensure version string has major, minor and micro components.
    while len(version.split('.')) < 3:
        version += '.0'

    return version

version = readGitVersion().replace('post', '')

# write the 'properties.yml' file
properties = {'plugin_name': plugin_name, 'package_name': package_name,
              'version': version}
with open('properties.yml', 'w') as f:
    f.write(yaml.dump(properties))

# create the tar.gz plugin archive
with tarfile.open("%s-%s.tar.gz" % (package_name, version), "w:gz") as tar:
    for name in ['__init__.py', 'feedback.py', 'glade', 'properties.yml',
                 'hooks', 'on_plugin_install.py', 'requirements.txt', 'wizards.py',
                 'COPYING', 'CHANGELOG']:
        tar.add(name)
