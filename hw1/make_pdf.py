"""Build HW1.pdf from README.md (GitHub link on page 1) using headless Chrome."""
import markdown, subprocess, os
body = markdown.markdown(open('README.md').read(), extensions=['tables', 'fenced_code'])
link = '<p><b>Code:</b> <a href="https://github.com/XinyiLiu777/CS6180/tree/main/hw1">https://github.com/XinyiLiu777/CS6180/tree/main/hw1</a></p>'
body = body.replace('</h1>', '</h1>' + link, 1)
css = ('body{font-family:Helvetica,Arial,sans-serif;font-size:12px;line-height:1.5;max-width:780px;margin:auto}'
       'img{max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #999;padding:3px 6px}'
       'pre{background:#f4f4f4;padding:6px;font-size:10px;white-space:pre-wrap}h2{page-break-before:auto;border-bottom:1px solid #ccc}')
open('HW1.html', 'w').write(f'<html><head><meta charset="utf-8"><style>{css}</style></head><body>{body}</body></html>')
subprocess.run(['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', '--headless', '--disable-gpu',
                '--no-pdf-header-footer', f'--print-to-pdf={os.path.abspath("HW1.pdf")}', f'file://{os.path.abspath("HW1.html")}'],
               check=True, capture_output=True)
os.remove('HW1.html')
print('wrote HW1.pdf')
