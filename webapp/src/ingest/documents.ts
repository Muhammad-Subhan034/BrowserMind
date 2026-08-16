// Document loading: plain text and Markdown are read directly; PDF text
// extraction uses pdfjs-dist (Mozilla's PDF.js) since reimplementing a PDF
// parser is out of scope for a project about GPU compute kernels -- this
// is the one place the app leans on an existing library rather than
// hand-written code, and deliberately so.

export interface LoadedDocument {
  name: string;
  text: string;
}

export async function loadFile(file: File): Promise<LoadedDocument> {
  const ext = file.name.split(".").pop()?.toLowerCase();
  if (ext === "pdf") {
    const text = await extractPdfText(file);
    return { name: file.name, text };
  }
  const text = await file.text();
  return { name: file.name, text };
}

async function extractPdfText(file: File): Promise<string> {
  const pdfjs = await import("pdfjs-dist");
  pdfjs.GlobalWorkerOptions.workerSrc = new URL(
    "pdfjs-dist/build/pdf.worker.mjs",
    import.meta.url,
  ).toString();

  const buffer = await file.arrayBuffer();
  const doc = await pdfjs.getDocument({ data: buffer }).promise;

  const pages: string[] = [];
  for (let pageNum = 1; pageNum <= doc.numPages; pageNum++) {
    const page = await doc.getPage(pageNum);
    const content = await page.getTextContent();
    const strings = content.items.map((item) => ("str" in item ? item.str : ""));
    pages.push(strings.join(" "));
  }
  return pages.join("\n\n");
}
