import re

word_classes = {}
word_classes["twoDigitNum"] = r'^\d\d$'
word_classes["fourDigitNum"] = r'^\d\d\d\d$'
word_classes["containsDigitAndDash"] = r'^(\d+-+\d+)+$'
word_classes["containsDigitsAndSlashWithEscape"] = r'^(\d+\\\/+\d+)+$'
word_classes["containsDigitsAndSlash"] = r'^(\d+\/+\d+)+$'
word_classes["containsDigitsAndPeriod"] = r'^\d+\.\d+$'
word_classes["containsDigitsAndCommas"] = r'^(\d+,*)+$'
word_classes["othernum"] = r'^\d+$'
word_classes["allCapsPlural"] = r'^[A-Z]+s$'
word_classes["allCaps"] = r'^[A-Z]+$'
word_classes["capPeriod"] = r'^([A-Z]+\.)+$'
word_classes["dashedWord"] = r'^\w+-\w+$'
word_classes["initCap"] = r'^[A-Z]\w+$'
word_classes["initCap-p"] = r'^[A-Z]\w+s$'
word_classes["$lowercase-p$"] = r'^[a-z]+s$'
word_classes["$lowercase-s$"] = r'^[a-z]+$'


word_classes_test = {}
word_classes_test["twoDigitNum"] = r'^\d\d$'
word_classes_test["fourDigitNum"] = r'^\d\d\d\d$'
word_classes_test["containsDigitAndDash"] = r'^(\d+-+\d+)+$'
word_classes_test["containsDigitsAndSlashWithEscape"] = r'^(\d+\\\/+\d+)+$'
word_classes_test["containsDigitsAndSlash"] = r'^(\d+\/+\d+)+$'
word_classes_test["containsDigitsAndPeriod"] = r'^\d+\.\d+$'
word_classes_test["containsDigitsAndCommas"] = r'^(\d+,*)+$'
word_classes_test["capPeriod"] = r'^([A-Z]+\.)+$'
word_classes_test["othernum"] = r'^\d+$'
word_classes_test["allCapsPlural"] = r'^[A-Z]+s$'
word_classes_test["allCaps"] = r'^[A-Z]+$'
word_classes_test["dashedWord"] = r'^\w+-\w+$'
word_classes_test["initCap"] = r'^[A-Z]\w+$'
word_classes_test["initCap-p"] = r'^[A-Z]\w+s$'

def get_word_class(word):
    for clz in word_classes.keys():
        if re.match(word_classes[clz], word):
            return clz
    return "$other$"

def get_word_class_test(word):
    for clz in word_classes_test.keys():
        if re.match(word_classes_test[clz], word):
            return clz
    return word