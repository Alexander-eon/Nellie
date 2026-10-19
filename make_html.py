import markdown

with open('ARTICLE_2.md', 'r', encoding='utf-8') as f:
    text = f.read()

# Вставляем график перед разделом 3.7
if "#### 3.7." in text:
    text = text.replace("#### 3.7.", 
        "<div style='text-align:center; margin: 20px 0;'>\n"
        "<img src='language_evolution_plots.png' style='max-width:100%;'>\n"
        "<p><i>Рис. 1. Слева: эволюция языка и достижение предела Шеннона. По центру: островная модель и рождение диалектов. Справа: горизонтальный перенос генов (гибридная сила и депрессия).</i></p>\n"
        "</div>\n\n#### 3.7.")

html_content = markdown.markdown(text, extensions=['tables', 'fenced_code'])

styled_html = f"""
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>От Хебба к Predictive Coding</title>
    <style>
        body {{ font-family: 'Georgia', serif; line-height: 1.6; max-width: 900px; margin: 40px auto; padding: 20px; color: #333; }}
        h1 {{ text-align: center; font-size: 2em; margin-bottom: 10px; }}
        h2 {{ color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 5px; margin-top: 30px; }}
        h3 {{ color: #34495e; margin-top: 25px; }}
        h4 {{ color: #16a085; margin-top: 20px; }}
        p {{ text-align: justify; margin-bottom: 15px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 0.9em; }}
        th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
        th {{ background-color: #f8f9fa; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        code {{ background-color: #f4f4f4; padding: 2px 5px; border-radius: 3px; }}
        img {{ border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
        @media print {{ body {{ margin: 0; padding: 20px; }} h2 {{ page-break-after: avoid; }} table, img {{ page-break-inside: avoid; }} }}
    </style>
</head>
<body>
{html_content}
</body>
</html>
"""

with open('ARTICLE_2.html', 'w', encoding='utf-8') as f:
    f.write(styled_html)

print("HTML-файл создан: ARTICLE_2.html")
