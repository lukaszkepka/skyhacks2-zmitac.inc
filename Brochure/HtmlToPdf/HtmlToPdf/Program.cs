using Codaxy.WkHtmlToPdf;
using Newtonsoft.Json;
using Spire.Pdf;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HtmlToPdf
{
    class Program
    {


        //public class Item
        //{
        //    public string text;
        //    public string image;

        //}


        static void Main(string[] args)
        {

            LoadJson();
            combinePdf();

            //PdfConvert.ConvertHtmlToPdf(new PdfDocument
            //{
            //    Url = "template.html",
            //    HeaderLeft = "[title]",
            //    HeaderRight = "[date] [time]",
            //    FooterCenter = "Page [page] of [topage]"

            //}, new PdfOutput
            //{
            //    OutputFilePath = "wkhtmltopdf-page.pdf"
            //});

        }

        private static void combinePdf()
        {
            String[] files = new String[] { "pdf1.pdf", "pdf2.pdf", "pdf3.pdf" };
            string outputFile = "result.pdf";
            Spire.Pdf.PdfDocumentBase doc = Spire.Pdf.PdfDocument.MergeFiles(files);
            doc.Save(outputFile, FileFormat.PDF);
        }

        public static void LoadJson()
        {

            using (StreamReader r = new StreamReader("data.json"))
            {
                string json = r.ReadToEnd();
                dynamic array = JsonConvert.DeserializeObject(json);

                var file1 = array["test_1"];
                writehtmlandpdf(file1, 1);
                var file2 = array["test_2"];
                writehtmlandpdf(file2, 2);
                var file3 = array["test_3"];
                writehtmlandpdf(file3, 3);


                //List<Item> items = JsonConvert.DeserializeObject<List<Item>>(json);
            }
        }

        public static void writehtmlandpdf(Newtonsoft.Json.Linq.JArray a,int number)
        {
            int i = 1;
            string text = File.ReadAllText("ultra_template.html");
            foreach (var aa in a){

                text = text.Replace("1\\img"+i , number+"\\"+aa["image"].ToString());
                text = text.Replace("desc" + i, aa["text"].ToString());
                File.WriteAllText("template+" + number + ".html", text);
                i++;
            }



            PdfConvert.ConvertHtmlToPdf(new Codaxy.WkHtmlToPdf.PdfDocument
            {
                Url = "template+"+number+".html",
                HeaderLeft = "[title]",
                HeaderRight = "[date] [time]",
                FooterCenter = "Page [page] of [topage]"

            }, new PdfOutput
            {
                OutputFilePath = "pdf"+number+".pdf"
            });


        }
    }
}
