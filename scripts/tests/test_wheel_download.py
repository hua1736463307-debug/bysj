import hashlib
import importlib.util
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).parents[1]))
import download_report_wheels as downloader


def test_range_resume_and_full_digest(tmp_path,monkeypatch):
    content=b'first-range-last-range'
    filename='example-1.0-py3-none-any.whl'
    (tmp_path/(filename+'.part')).write_bytes(content[:6])
    report={'environment':{'sys_platform':'linux','python_version':'3.12'},'install':[
        {'metadata':{'name':'example','version':'1.0'},'download_info':{'url':'https://files.pythonhosted.org/'+filename,
            'archive_info':{'hashes':{'sha256':hashlib.sha256(content).hexdigest()}}}}]}
    ranges=[]
    class Response:
        status=206
        headers={'Content-Range':f'bytes 6-{len(content)-1}/{len(content)}'}
        def read(self,*args):return content[6:]
        def __enter__(self):return self
        def __exit__(self,*args):pass
    def open_url(request,**kwargs):
        ranges.append(request.get_header('Range'));return Response()
    monkeypatch.setattr(downloader.urllib.request,'urlopen',open_url)
    downloader.download(report,tmp_path)
    assert ranges==['bytes=6-4194309']
    assert (tmp_path/filename).read_bytes()==content
    downloader.download(report,tmp_path)
    assert len(ranges)==1
