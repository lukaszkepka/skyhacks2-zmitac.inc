using Codaxy.WkHtmlToPdf;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HtmlToPdf
{
    class Program
    {
        static void Main(string[] args)
        {

            PdfConvert.ConvertHtmlToPdf(new PdfDocument
            {
                Url = "C:\\Users\\matpa\\source\\repos\\HtmlToPdf\\HtmlToPdf\\template.html",
                HeaderLeft = "[title]",
                HeaderRight = "[date] [time]",
                FooterCenter = "Page [page] of [topage]"

            }, new PdfOutput
            {
                OutputFilePath = "wkhtmltopdf-page.pdf"
            });

        }
    }
}
