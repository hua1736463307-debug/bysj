"""Create an explicit legacy retirement inventory; never delete files."""
import ast
import json
from pathlib import Path
import re
import posixpath
import subprocess
from legacy_schema import REPOSITORY, BASELINE

FRONTEND = {
    'frontend/src/api/api.js', 'frontend/src/utils/api.js',
    'frontend/src/components/BatchInferencePanel.vue', 'frontend/src/components/PoseImageTestPanel.vue',
    'frontend/src/components/PoseTrainingPanel.vue', 'frontend/src/composables/useAnnotationHistory.js',
    'frontend/src/composables/useTrainingJobs.js', 'frontend/src/utils/annotationLabeling.mjs',
    'frontend/src/utils/imageAugmentation.mjs', 'frontend/src/utils/modelCapabilities.mjs',
    'frontend/src/utils/poseSubmission.test.mjs', 'frontend/src/utils/poseTraining.mjs',
    'frontend/src/utils/poseTraining.test.mjs',
}


def audit():
    tracked = subprocess.check_output(['git','-C',str(REPOSITORY),'ls-files','-z']).decode().split('\0')
    candidates = set(FRONTEND)
    for name in tracked:
        parts = Path(name).parts
        if len(parts) == 2 and parts[0] == 'backend' and name.endswith('.py') and parts[-1] != 'start_api.py':
            candidates.add(name)
        if len(parts) > 2 and parts[0] == 'backend' and parts[1] in {'behavior','face_capture','inference_service','pose_training','utils','sql'}:
            candidates.add(name)
        if len(parts) == 3 and parts[:2] == ('backend','tests') and parts[-1] not in {'conftest.py','test_app_factory_auth.py'}:
            candidates.add(name)
    texts = {name:(REPOSITORY/name).read_text(encoding='utf-8-sig') for name in tracked
             if name and (REPOSITORY/name).is_file() and Path(name).suffix in {'.py','.vue','.js','.mjs','.cjs','.yml','.yaml'} }
    module_files = {}
    for name in texts:
        if name.startswith('backend/') and name.endswith('.py'):
            module = name[len('backend/'):-3].replace('/', '.')
            if module.endswith('.__init__'): module = module[:-9]
            module_files[module] = name
    dependencies = {name:set() for name in texts}
    for name, content in texts.items():
        if name.endswith('.py'):
            for node in ast.walk(ast.parse(content, filename=name)):
                modules = []
                if isinstance(node, ast.Import): modules = [n.name for n in node.names]
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    if node.level:
                        package = name[len('backend/'):].split('/')[:-1]
                        module = '.'.join(package[:len(package)-node.level+1] + ([module] if module else []))
                    modules = [module, *(module+'.'+n.name for n in node.names)]
                for module in modules:
                    if module in module_files: dependencies[name].add(module_files[module])
        elif Path(name).suffix in {'.vue','.js','.mjs','.cjs'}:
            for source in re.findall(r'(?:from\s*|import\s*\(?|require\s*\()\s*[\'\"]([^\'\"]+)[\'\"]', content):
                if source.startswith('.'):
                    target = posixpath.normpath(posixpath.join(posixpath.dirname(name), source))
                    for candidate in [target, target+'.js', target+'.vue', target+'.mjs']:
                        if candidate in texts: dependencies[name].add(candidate)
    inventory = []
    for name in sorted(candidates):
        refs = sorted(other for other,values in dependencies.items() if other != name and name in values)
        external = [r for r in refs if r not in candidates]
        inventory.append({'file':name,'references':refs,'outside_retirement_set':external,
            'runtime_use':'legacy-only or unreachable from the v1 entry; verify external references before removal',
            'reason':'retire superseded industry/runtime path after v1 browser, comparison and migration gates'})
    return {'baseline':BASELINE,'mode':'inventory_only','candidate_count':len(inventory),'files':inventory}


if __name__ == '__main__':
    report = audit()
    output = REPOSITORY/'docs/refactor/execution/unblock/legacy-cleanup-inventory.json'
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(output),'count':report['candidate_count'],
        'external_references':{r['file']:r['outside_retirement_set'] for r in report['files'] if r['outside_retirement_set']}},ensure_ascii=False,indent=2))
