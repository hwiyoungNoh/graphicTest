#pragma once



// CResultsView form view

class CResultsView : public CFormView
{
	DECLARE_DYNCREATE(CResultsView)

protected:
	CResultsView();           // protected constructor used by dynamic creation
	virtual ~CResultsView();

public:
	void InsertText(CString text, COLORREF color, bool bold, bool italic);
	CRichEditCtrl m_editBox;
	enum { IDD = IDD_RESULTSVIEW };
#ifdef _DEBUG
	virtual void AssertValid() const;
#ifndef _WIN32_WCE
	virtual void Dump(CDumpContext& dc) const;
#endif
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support
	afx_msg void OnSize(UINT nType, int cx, int cy) ;

	DECLARE_MESSAGE_MAP()
};


