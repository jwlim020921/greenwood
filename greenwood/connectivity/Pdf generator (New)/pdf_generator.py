import jinja2
import pdfkit
def generatePdf(elephant_frequency:int, motorbike_frequency:int, gunshot_frequency:int, human_frequency:int, 
                logging_frequency:int, logging_occurences:int, poaching_occurences:int):

    elephant_f = elephant_frequency
    motorbike_f = motorbike_frequency
    gunshot_f = gunshot_frequency
    human_f = human_frequency
    logging_f = logging_frequency
    logging_o = logging_occurences
    poaching_o = poaching_occurences


    context = {'ef': elephant_f, 'mf': motorbike_f, 
            'gf': gunshot_f, 'hf': human_f, 'lf' : logging_f, 
            'lo' : logging_o, 'po' : poaching_o}

    template_loader = jinja2.FileSystemLoader('./')
    template_env = jinja2.Environment(loader=template_loader)

    file_template = r'pdf-template.html'
    template = template_env.get_template(file_template)
    output_text = template.render(context)

    config = pdfkit.configuration(wkhtmltopdf=r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe')
    pdfkit.from_string(output_text, 'generated_data.pdf',
    configuration=config, css = 'style.css')

generatePdf(15, 15, 6, 92, 58, 59, 83)