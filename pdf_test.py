from PyPDF2 import PdfFileReader, PdfFileMerger
merger = PdfFileMerger(strict=False)
merger.append(PdfFileReader("/home/saurabh/Downloads/pancard.pdf"), import_bookmarks=False)
with open("/home/saurabh/Downloads/new_pan.pdf", 'wb') as pdf_out:
    merger.write(pdf_out)