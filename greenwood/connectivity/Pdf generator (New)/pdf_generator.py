import jinja2
import pdfkit
import os
import sys
root_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(root_directory, 'import'))

class pdfgen:
	def __init__(self, elephant_f, motorbike_f, gunshot_f, human_f, logging_f, logging_o, poaching_o):

		self.elephant_f = elephant_f
		self.motorbike_f = motorbike_f
    		self.gunshot_f = gunshot_f
    		self.human_f = human_f
    		self.logging_f = logging_f
    		self.logging_o = logging_o
    		self.poaching_o = poaching_o


	def pdfgenerator(greenwood):
		context = {'ef': elephant_f, 'mf': motorbike_f, 'gf': gunshot_f, 'hf': human_f, 'lf' : logging_f, 'lo' : logging_o, 'po' : poaching_o}

    		template_loader = jinja2.FileSystemLoader('./')
    		template_env = jinja2.Environment(loader=template_loader)

    		file_template = r'pdf-template.html'
    		template = template_env.get_template(file_template)
    		output_text = template.render(context)

    		config = pdfkit.configuration(wkhtmltopdf=r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe')
    		pdfkit.from_string(output_text, 'generated_data.pdf',
    		configuration = config, css = 'style.css')

generatePdf = pdfgen(15, 15, 6, 92, 58, 59, 83)
generatePdf.self()
