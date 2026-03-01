// DataView.cpp : implementation file
//

#include "stdafx.h"
#include "RemoteCommunication.h"
#include "DataView.h"


// CDataView

IMPLEMENT_DYNCREATE(CDataView, CFormView)

CDataView::CDataView()
	: CFormView(CDataView::IDD)
{
	m_measurementIndex = 0;

}

CDataView::~CDataView()
{
}

void CDataView::DoDataExchange(CDataExchange* pDX)
{
	CFormView::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_TAB_DATA, m_TabCtrl);
	DDX_Control(pDX, IDC_LIST_SPECTRUM, m_spectrumListCtrl);
	DDX_Control(pDX, IDC_LIST_GENERAL, m_generalListCtrl);
	DDX_Control(pDX, IDC_LIST_TEMPORAL, m_temporalListCtrl);
	DDX_Control(pDX, IDC_LIST_FLICKER, m_flickerListCtrl);
	DDX_Control(pDX, IDC_LIST_RESPONSETIME, m_responseTimeListCtrl);
	DDX_Control(pDX, IDC_LIST_CALCULATION, m_calculationListCtrl);
}

BEGIN_MESSAGE_MAP(CDataView, CFormView)
	ON_WM_SIZE()
	ON_WM_CREATE()
	ON_NOTIFY(TCN_SELCHANGE, IDC_TAB_DATA, &CDataView::OnTcnSelchangeTabData)
	ON_NOTIFY(TCN_SELCHANGING, IDC_TAB_DATA, &CDataView::OnTcnSelchangingTabData)
	ON_NOTIFY(NM_CUSTOMDRAW, IDC_LIST_SPECTRUM, &CDataView::OnNMCustomdrawList)
	ON_NOTIFY(NM_CUSTOMDRAW, IDC_LIST_GENERAL, &CDataView::OnNMCustomdrawList)
	ON_NOTIFY(NM_CUSTOMDRAW, IDC_LIST_TEMPORAL, &CDataView::OnNMCustomdrawList)
	ON_NOTIFY(NM_CUSTOMDRAW, IDC_LIST_FLICKER, &CDataView::OnNMCustomdrawList)
	ON_NOTIFY(NM_CUSTOMDRAW, IDC_LIST_RESPONSETIME, &CDataView::OnNMCustomdrawList)
	ON_NOTIFY(NM_CUSTOMDRAW, IDC_LIST_CALCULATION, &CDataView::OnNMCustomdrawList)
END_MESSAGE_MAP()


// CDataView diagnostics

#ifdef _DEBUG
void CDataView::AssertValid() const
{
	CFormView::AssertValid();
}

#ifndef _WIN32_WCE
void CDataView::Dump(CDumpContext& dc) const
{
	CFormView::Dump(dc);
}
#endif
#endif //_DEBUG


// CDataView message handlers

void CDataView::OnSize(UINT nType, int cx, int cy)
{
	CFormView::OnSize(nType, cx, cy);

	// TODO: Add your message handler code here
	if(m_TabCtrl.m_hWnd)
	{
		CRect rc;
		CRect tabRect, itemRect;
		int nX, nY, nXc, nYc;
		GetClientRect(&rc);
		m_TabCtrl.MoveWindow(&rc);


		int index = m_TabCtrl.GetCurSel();
		m_TabCtrl.GetClientRect(&tabRect);
		m_TabCtrl.GetItemRect(0, &itemRect);
		nX = tabRect.left;
		nY = tabRect.top + itemRect.Height() + 1;
		nXc = tabRect.right;
		nYc = tabRect.bottom - itemRect.Height();

		if(m_generalListCtrl.m_hWnd)
			m_generalListCtrl.MoveWindow(nX, nY, nXc, nYc, 0);

		if(m_spectrumListCtrl.m_hWnd)
			m_spectrumListCtrl.MoveWindow(nX, nY, nXc, nYc, 0);

		if(m_temporalListCtrl.m_hWnd)
			m_temporalListCtrl.MoveWindow(nX, nY, nXc, nYc, 0);

		if(m_flickerListCtrl.m_hWnd)
			m_flickerListCtrl.MoveWindow(nX, nY, nXc-300, nYc, 0);

		if(m_flickerSetupCtrl.m_hWnd)
			m_flickerSetupCtrl.MoveWindow(nXc-300, nY, nXc, nYc, 0);
				
		if(m_responseTimeListCtrl.m_hWnd)
			m_responseTimeListCtrl.MoveWindow(nX, nY, nXc-300, nYc, 0);

		if(m_responseTimeSetupCtrl.m_hWnd)
			m_responseTimeSetupCtrl.MoveWindow(nXc-300, nY, nXc, nYc, 0);
		
		if(m_calculationListCtrl.m_hWnd)
			m_calculationListCtrl.MoveWindow(nX, nY, nXc-300, nYc, 0);

		if(m_calculationSetupCtrl.m_hWnd)
			m_calculationSetupCtrl.MoveWindow(nXc-300, nY, nXc, nYc, 0);
		


		InvalidateRect(rc);
	}

}

int CDataView::OnCreate( LPCREATESTRUCT lpCreateStruct )
{
	int iResult = CFormView::OnCreate(lpCreateStruct);

	if(iResult == 0)
	{
		m_boldFont.CreateFont(14, 0, 0, 0, FW_BOLD, 
						FALSE, FALSE, FALSE, 
						DEFAULT_CHARSET, 
						OUT_DEFAULT_PRECIS, 
						CLIP_DEFAULT_PRECIS, 
						DEFAULT_QUALITY, 
						DEFAULT_PITCH, 
						NULL);
		
	}
	
	VERIFY(m_flickerSetupCtrl.Create(CFlickerSetupDlg::IDD, this));
	VERIFY(m_responseTimeSetupCtrl.Create(CResponseTimeSetupDlg::IDD, this));
	VERIFY(m_calculationSetupCtrl.Create(CCalculationSetupDlg::IDD, this));

	return iResult;
}

void CDataView::OnInitialUpdate()
{
	CFormView::OnInitialUpdate();

	DWORD dwExStyle= m_TabCtrl.GetExtendedStyle();
	m_TabCtrl.SetExtendedStyle(dwExStyle | TCS_EX_FLATSEPARATORS);

	if(m_TabCtrl.GetItemCount() == 0)
	{
		int tabPosition = m_TabCtrl.InsertItem(TCIF_PARAM | TCIF_TEXT, 0, _T("General"), 0, 0);
		tabPosition = m_TabCtrl.InsertItem(TCIF_PARAM | TCIF_TEXT, 1, _T("Spectrum"), 0, 1);
		tabPosition = m_TabCtrl.InsertItem(TCIF_PARAM | TCIF_TEXT, 2, _T("Temporal"), 0, 2);
		tabPosition = m_TabCtrl.InsertItem(TCIF_PARAM | TCIF_TEXT, 3, _T("Flicker"), 0, 3);
		tabPosition = m_TabCtrl.InsertItem(TCIF_PARAM | TCIF_TEXT, 4, _T("Response Time"), 0, 4);
		tabPosition = m_TabCtrl.InsertItem(TCIF_PARAM | TCIF_TEXT, 5, _T("Calculations"), 0, 5);

		m_spectrumListCtrl.SetExtendedStyle(m_spectrumListCtrl.GetStyle() | LVS_EX_FULLROWSELECT);
		m_generalListCtrl.SetExtendedStyle(m_generalListCtrl.GetStyle() | LVS_EX_FULLROWSELECT);
		m_temporalListCtrl.SetExtendedStyle(m_temporalListCtrl.GetStyle() | LVS_EX_FULLROWSELECT);
		m_flickerListCtrl.SetExtendedStyle(m_flickerListCtrl.GetStyle() | LVS_EX_FULLROWSELECT);
		m_responseTimeListCtrl.SetExtendedStyle(m_responseTimeListCtrl.GetStyle() | LVS_EX_FULLROWSELECT);
		m_calculationListCtrl.SetExtendedStyle(m_calculationListCtrl.GetStyle() | LVS_EX_FULLROWSELECT);
	
	
		m_generalListCtrl.InsertColumn(0, _T(""), LVCFMT_LEFT, 200);
		m_spectrumListCtrl.InsertColumn(0, _T(""), LVCFMT_LEFT, 200);
		m_temporalListCtrl.InsertColumn(0, _T(""), LVCFMT_LEFT, 200);
		m_flickerListCtrl.InsertColumn(0, _T(""), LVCFMT_LEFT, 200);
		m_responseTimeListCtrl.InsertColumn(0, _T(""), LVCFMT_LEFT, 200);
		m_calculationListCtrl.InsertColumn(0, _T(""), LVCFMT_LEFT, 200);

		
		m_generalListCtrl.ShowWindow(SW_SHOW);
		m_spectrumListCtrl.ShowWindow(SW_HIDE);
		m_temporalListCtrl.ShowWindow(SW_HIDE);
		m_flickerListCtrl.ShowWindow(SW_HIDE);
		m_flickerSetupCtrl.ShowWindow(SW_HIDE);
		m_responseTimeListCtrl.ShowWindow(SW_HIDE);
		m_responseTimeSetupCtrl.ShowWindow(SW_HIDE);
		m_calculationListCtrl.ShowWindow(SW_HIDE);
		m_calculationSetupCtrl.ShowWindow(SW_HIDE);

		m_generalListCtrl.SetFocus();
	}
}

int CDataView::getCurrentTabID()
{
	int tabID = -1;
	if(m_TabCtrl.GetItemCount() > 0)
	{
		int nIndex= m_TabCtrl.GetCurSel();
		if(nIndex >= 0)
		{
			TC_ITEM TabCtrlItem;
			TabCtrlItem.mask = TCIF_PARAM;
			TabCtrlItem.lParam = NULL;
			if(m_TabCtrl.GetItem(nIndex, &TabCtrlItem))
				tabID = (UINT)TabCtrlItem.lParam;
		}
	}
	return tabID;
}

void CDataView::OnTcnSelchangeTabData(NMHDR *pNMHDR, LRESULT *pResult)
{
	// TODO: Add your control notification handler code here

	int tabID = getCurrentTabID();
	CWnd* pTabPage = NULL;
	
	switch(tabID)
	{
	case 0:
		pTabPage = &m_generalListCtrl;
		break;
	case 1:
		pTabPage = &m_spectrumListCtrl;
		break;
	case 2:
		pTabPage = &m_temporalListCtrl;
		break;
	case 3:
		pTabPage = &m_flickerListCtrl;
		m_flickerSetupCtrl.ShowWindow(SW_SHOW);
		break;
	case 4:
		pTabPage = &m_responseTimeListCtrl;
		m_responseTimeSetupCtrl.ShowWindow(SW_SHOW);
		break;
	case 5:
		pTabPage = &m_calculationListCtrl;
		m_calculationSetupCtrl.ShowWindow(SW_SHOW);
		break;

	}

	if(pTabPage)
	{
		pTabPage->ShowWindow(SW_SHOW);
		pTabPage->BringWindowToTop();
		pTabPage->SetFocus();

		*pResult = 0;
	}
}

void CDataView::SetTabRectangle()
{
		
}

void CDataView::OnTcnSelchangingTabData(NMHDR *pNMHDR, LRESULT *pResult)
{
	// TODO: Add your control notification handler code here

	int tabID = getCurrentTabID();
	switch(tabID)
	{
	case 0: // General
		m_generalListCtrl.ShowWindow(SW_HIDE);
		m_flickerSetupCtrl.ShowWindow(SW_HIDE);
		m_responseTimeSetupCtrl.ShowWindow(SW_HIDE);
		break;
	case 1: // Spectral
		m_spectrumListCtrl.ShowWindow(SW_HIDE);
		m_responseTimeSetupCtrl.ShowWindow(SW_HIDE);
		m_flickerSetupCtrl.ShowWindow(SW_HIDE);
		break;
	case 2: // Temporal
		m_temporalListCtrl.ShowWindow(SW_HIDE);
		m_flickerSetupCtrl.ShowWindow(SW_HIDE);
		m_responseTimeSetupCtrl.ShowWindow(SW_HIDE);
		break;
	case 3: // Flicker
		m_flickerListCtrl.ShowWindow(SW_HIDE);
		m_flickerSetupCtrl.ShowWindow(SW_HIDE);
		m_responseTimeSetupCtrl.ShowWindow(SW_HIDE);
		break;		
	case 4: // Response Time
		m_responseTimeListCtrl.ShowWindow(SW_HIDE);
		m_flickerSetupCtrl.ShowWindow(SW_HIDE);
		m_responseTimeSetupCtrl.ShowWindow(SW_HIDE);
		break;	
	case 5: // Calculation
		m_calculationListCtrl.ShowWindow(SW_HIDE);
		m_calculationSetupCtrl.ShowWindow(SW_HIDE);
		break;
	}
	*pResult = 0;
}


void CDataView::UpdateText(CListCtrl &list, const CString& type, const CString& data, int measurement)
{
	LVFINDINFO info;
	LV_ITEM item = {0};
	int index;

	info.flags = LVFI_STRING;
	info.psz = (LPTSTR)(LPCTSTR)type;

	index = list.FindItem(&info);

	if(index == -1) // add 
	{	
		item.iItem = list.GetItemCount();
		item.iSubItem = 0;
		item.mask = LVIF_TEXT;
		item.pszText = (LPTSTR)(LPCTSTR)type;
		item.cchTextMax	= type.GetLength();
		index = list.InsertItem(&item);

		item.iItem = index;
		item.iSubItem = measurement;
		item.mask = LVIF_TEXT;
		item.pszText = (LPTSTR)(LPCTSTR)data;
		item.cchTextMax	= data.GetLength();

		index = list.SetItem(&item);
	}
	else	// update
	{
		item.iItem = index;
		item.iSubItem = measurement;
		item.mask = LVIF_TEXT;
		item.pszText = (LPTSTR)(LPCTSTR)data;
		item.cchTextMax	= data.GetLength();
		
		index = list.SetItem(&item);
	}
}

void CDataView::AddMeasurement()
{
	m_measurementIndex++;
	CString name;
	name.Format(_T("M%d"), m_measurementIndex); 
	m_generalListCtrl.InsertColumn(m_measurementIndex, name, LVCFMT_LEFT, 200);
	m_spectrumListCtrl.InsertColumn(m_measurementIndex, name, LVCFMT_LEFT, 200);
	m_temporalListCtrl.InsertColumn(m_measurementIndex, name, LVCFMT_LEFT, 200);
	m_flickerListCtrl.InsertColumn(m_measurementIndex, name, LVCFMT_LEFT, 200);
	m_responseTimeListCtrl.InsertColumn(m_measurementIndex, name, LVCFMT_LEFT, 200);
	m_calculationListCtrl.InsertColumn(m_measurementIndex, name, LVCFMT_LEFT, 200);

}

void CDataView::AddMeasurementData(const CString& type, const CString& data)
{		
	UpdateText(m_generalListCtrl, type, data, m_measurementIndex);	
}

void CDataView::AddSpectralData(const CString& type, const CString& data)
{
	UpdateText(m_spectrumListCtrl, type, data, m_measurementIndex);
}

void CDataView::AddTemporalData(const CString& type, const CString& data)
{
	UpdateText(m_temporalListCtrl, type, data, m_measurementIndex);
}

void CDataView::AddFlickerData(const CString& type, const CString& data)
{
	UpdateText(m_flickerListCtrl, type, data, m_measurementIndex);
}

void CDataView::AddResponseTimeData(const CString& type, const CString& data)
{
	UpdateText(m_responseTimeListCtrl, type, data, m_measurementIndex);
}

void CDataView::AddCalculationData(const CString& type, const CString& data)
{
	UpdateText(m_calculationListCtrl, type, data, m_measurementIndex);
}


void CDataView::OnNMCustomdrawList(NMHDR *pNMHDR, LRESULT *pResult)
{
	NMLVCUSTOMDRAW* pLVCD = reinterpret_cast<NMLVCUSTOMDRAW*>(pNMHDR);
	// TODO: Add your control notification handler code here
	 *pResult = CDRF_DODEFAULT;

	 if ( CDDS_PREPAINT == pLVCD->nmcd.dwDrawStage )
	 {
		 *pResult = CDRF_NOTIFYITEMDRAW;
	 }
	 else if ( CDDS_ITEMPREPAINT == pLVCD->nmcd.dwDrawStage )
	 {
		 // This is the prepaint stage for an item. Here's where we set the
		 // item's text color. Our return value will tell Windows to draw the
		 // item itself, but it will use the new color we set here.
		 // We'll cycle the colors through red, green, and light blue.

		 if ( (pLVCD->nmcd.dwItemSpec % 2) == 0 )
			 pLVCD->clrTextBk = RGB(231, 237, 246);

		 /*if (pLVCD->iPartId == 1)
		 {
			 SelectObject(pLVCD->nmcd.hdc, (HFONT)m_boldFont);
			*pResult = CDRF_NEWFONT;
		 }*/


		 // Tell Windows to paint the control itself.
		 *pResult |= CDRF_DODEFAULT;
	 }
}
