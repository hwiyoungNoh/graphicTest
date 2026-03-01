#pragma once
#include "afxcmn.h"

#include "FlickerSetupDlg.h"
#include "ResponseTimeSetupDlg.h"
#include "CalculationSetupDlg.h"

// CDataView form view

class CDataView : public CFormView
{
	DECLARE_DYNCREATE(CDataView)

protected:
	CDataView();           // protected constructor used by dynamic creation
	virtual ~CDataView();


public:
	enum { IDD = IDD_DATAVIEW };
#ifdef _DEBUG
	virtual void AssertValid() const;
#ifndef _WIN32_WCE
	virtual void Dump(CDumpContext& dc) const;
#endif
#endif
	void AddMeasurement();
	void AddMeasurementData(const CString& type, const CString& data);
	void AddSpectralData(const CString& type, const CString& data);
	void AddTemporalData(const CString& type, const CString& data);
	void AddFlickerData(const CString& type, const CString& data);
	void AddResponseTimeData(const CString& type, const CString& data);
	void AddCalculationData(const CString& type, const CString& data);


protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support
	afx_msg void OnSize(UINT nType, int cx, int cy) ;
	afx_msg int OnCreate(LPCREATESTRUCT lpCreateStruct);
	void OnInitialUpdate();
	int getCurrentTabID();
	void SetTabRectangle();
	void UpdateText(CListCtrl &list, const CString& type, const CString& data, int measurement);
	DECLARE_MESSAGE_MAP()

	CTabCtrl m_TabCtrl;
	CFont m_boldFont;
	int m_measurementIndex;
public:
	afx_msg void OnTcnSelchangeTabData(NMHDR *pNMHDR, LRESULT *pResult);
	afx_msg void OnTcnSelchangingTabData(NMHDR *pNMHDR, LRESULT *pResult);
	afx_msg void OnNMCustomdrawList(NMHDR *pNMHDR, LRESULT *pResult);
	
	CListCtrl m_generalListCtrl;
	CListCtrl m_spectrumListCtrl;
	CListCtrl m_temporalListCtrl;
	CListCtrl m_flickerListCtrl;
	CFlickerSetupDlg m_flickerSetupCtrl;
	CListCtrl m_responseTimeListCtrl;
	CListCtrl m_calculationListCtrl;
	CResponseTimeSetupDlg m_responseTimeSetupCtrl;
	CCalculationSetupDlg m_calculationSetupCtrl;
};


