import json
import pandas as pd
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak

json_file_path = "test_results.json"
pdf_file_path = "test_results_summary_2.pdf"


with open(json_file_path, 'r') as f:
    test_results = json.load(f)


df = pd.DataFrame(test_results)


df_ok = df[df['nnz_difference'] == False]
df_not_ok = df[df['nnz_difference'] == True]


def generate_pdf(df_ok, df_not_ok, pdf_file_path):
    doc = SimpleDocTemplate(pdf_file_path, pagesize=letter)
    elements = []
    

    styles = getSampleStyleSheet()
    title_style = styles['Title']
    normal_style = styles['Normal']


    title = Paragraph("Test Results Summary", title_style)
    elements.append(title)
    elements.append(Spacer(1, 12))
    

    doc.pagesize = landscape(letter)
    
    title_ok = Paragraph("Successful Test Cases", styles['Heading2'])
    elements.append(title_ok)
    col_widths = [40, 40, 40, 40, 50, 40, 100, 60, 60, 60]

    data_ok = [df_ok.columns.tolist()] + df_ok.values.tolist()
    table_ok = Table(data_ok,colWidths=col_widths)
    table_ok.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.green),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8), 
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(table_ok)
    elements.append(Spacer(1, 12))


    title_not_ok = Paragraph("Failed Test Cases", styles['Heading2'])
    elements.append(title_not_ok)

 
    data_not_ok = [df_not_ok.columns.tolist()] + df_not_ok.values.tolist()
    table_not_ok = Table(data_not_ok,colWidths=col_widths)
    table_not_ok.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.red),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8), 
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
   
    ]))
    elements.append(table_not_ok)
    elements.append(Spacer(1, 12))
    
 
    elements.append(PageBreak())
    doc.pagesize = letter

    for index, row in df_not_ok.iterrows():
        diff_text = f"""<br/><br/><strong>Difference Detected:</strong><br/>
                        Batch Size: {row['batch_size']}<br/>
                        Input Channels: {row['in_channels']}<br/>
                        Output Channels: {row['out_channels']}<br/>
                        Kernel Size: {row['kernel_size']}<br/>
                        Stride: {row['stride']}<br/>
                        Padding: {row['padding']}<br/>
                        Output Sum Difference: {row['output_sum_diff']}<br/>
                        nnz Cpp Result: {row['nnz_cpp_result']}<br/>
                        nnz Torch Eval: {row['nnz_torch_eval']}<br/>
                        """
        elements.append(Paragraph(diff_text, normal_style))

 
    doc.build(elements)


generate_pdf(df_ok, df_not_ok, pdf_file_path)

print(f"PDF report generated at {pdf_file_path}")
