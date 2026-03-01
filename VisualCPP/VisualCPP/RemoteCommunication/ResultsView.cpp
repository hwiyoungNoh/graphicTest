// ResultsView.cpp : implementation file
//

#include "stdafx.h"
#include "RemoteCommunication.h"
#include "ResultsView.h"


// CResultsView

IMPLEMENT_DYNCREATE(CResultsView, CFormView)

CResultsView::CResultsView()
	: CFormView(CResultsView::IDD)
{

}

CResultsView::~CResultsView()
{
}

void CResultsView::DoDataExchange(CDataExchange* pDX)
{
	CFormView::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_RICHEDIT_RESULTS, m_editBox);
}

BEGIN_MESSAGE_MAP(CResultsView, CFormView)
	ON_WM_SIZE()
END_MESSAGE_MAP()


// CResultsView diagnostics

#ifdef _DEBUG
void CResultsView::AssertValid() const
{
	CFormView::AssertValid();
}

#ifndef _WIN32_WCE
void CResultsView::Dump(CDumpContext& dc) const
{
	CFormView::Dump(dc);
}
#endif
#endif //_DEBUG


// CResultsView message handlers

void CResultsView::OnSize(UINT nType, int cx, int cy)
{
	CFormView::OnSize(nType, cx, cy);

	// TODO: Add your message handler code here

	if(m_editBox.m_hWnd)
	{
		CRect rect;
		GetClientRect(&rect);	
		m_editBox.SetWindowPos(&m_editBox,
			0, 0,
			rect.right,
			rect.bottom,
			SWP_NOZORDER | SWP_SHOWWINDOW);
	}
}

void CResultsView::InsertText(CString text, COLORREF color, bool bold, bool italic)
{
	CHARFORMAT cf = {0};
    int txtLen = m_editBox.GetTextLengthEx(GTL_NUMCHARS);
 
	int oldLines = m_editBox.GetLineCount();
    cf.cbSize = sizeof(cf);
    cf.dwMask = (bold ? CFM_BOLD : 0) | (italic ? CFM_ITALIC : 0) | CFM_COLOR;
    cf.dwEffects = (bold ? CFE_BOLD : 0) | (italic ? CFE_ITALIC : 0) |~CFE_AUTOCOLOR;
    cf.crTextColor = color;

    m_editBox.SetSel(txtLen, -1); // Set the cursor to the end of the text area and deselect everything.
    m_editBox.ReplaceSel(text); // Inserts when nothing is selected.

    // Apply formating to the just inserted text.
	
    m_editBox.SetSel(txtLen, m_editBox.GetTextLengthEx(GTL_NUMCHARS));
    m_editBox.SetSelectionCharFormat(cf);

	int newLines = m_editBox.GetLineCount();
	m_editBox.LineScroll(newLines - oldLines,0); 

}
