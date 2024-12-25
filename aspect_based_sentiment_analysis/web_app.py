from flask import Flask
from flask import request
from jinja2 import Environment, PackageLoader
from aspect_based_sentiment_analysis.text_labeler import TextLabeler


text_labeler = TextLabeler()
app = Flask(__name__)


@app.route("/", methods=['GET'])
def index() -> str:
    """
    GET-controller method for start page / main form

    :return: rendered html code for main form with default settings
    """
    return get_html_form(unlabeled_text="", labeled_text="")


@app.route("/", methods=['POST'])
def label() -> str:
    """
    POST-controller method to process form input and label a review text within main form

    :return: rendered html code for main form with labeled text and prefilled unlabeled review text
    """
    unlabeled_text = request.form.get("unlabeled_text")
    return get_html_form(unlabeled_text=unlabeled_text, labeled_text=text_labeler.label(unlabeled_text))


def get_html_form(unlabeled_text: str, labeled_text) -> str:
    """
    Helper method to pass settings to jinja for parsing the template from templates/template.html

    :param unlabeled_text: review text to label (if empty, placeholder instruction is shown)
    :param labeled_text: labeled text to be shown in main form
    :return: rendered html code for main form using the parameters
    """
    environment = Environment(loader=PackageLoader('aspect_based_sentiment_analysis', 'templates'))
    return environment.get_template("template.html").render(
        unlabeled_text=unlabeled_text,
        labeled_text=labeled_text
    )


if __name__ == '__main__':
    app.run()
