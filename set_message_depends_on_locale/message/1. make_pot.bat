rem Run once per update source.
rem .pot file defines the target of string to translate.
rem .po file is the implementation of translation.
pybabel extract -F babel.cfg --no-wrap -o locales/messages.pot .
pybabel init -i locales/messages.pot --no-wrap -d locales -l ja
echo "生成された .po ファイルを翻訳してください。"
pause
